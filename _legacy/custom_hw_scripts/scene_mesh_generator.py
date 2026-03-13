"""
==========================================================================
  SCENE MESH GENERATOR – Wrapper de Cenas para Hunyuan3D-2
==========================================================================

Wrapper que adapta o Hunyuan3D-2 para processar cenários amplos (mapas,
paisagens) em vez de apenas objetos isolados. O pipeline padrão do
Hunyuan3D-2 remove o background e foca em objetos centrais, o que não
é ideal para mapas/mundos gerados pelo HunyuanWorld.

Adaptações principais:
  1. Background removal desabilitado para cenários
  2. Parâmetros ajustados para geometria horizontal (terrenos)
  3. Resolução de octree aumentada para capturar detalhes de mapas
  4. Export direto como .glb (pronto para Unreal/Unity)

Uso via terminal (cloud/GPU alugada):
    python scene_mesh_generator.py --input /tmp/world_to_mesh/session/world_image.png \
                                    --output /tmp/world_to_mesh/session/world_mesh.glb \
                                    --scene_mode

Uso programático (chamado pelo bridge.py):
    from scene_mesh_generator import SceneMeshGenerator
    gen = SceneMeshGenerator()
    result = gen.generate(image_path, output_path, scene_mode=True)
==========================================================================
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Adiciona o diretório atual ao PYTHONPATH para encontrar o pacote hy3dgen
current_dir = os.path.dirname(os.path.abspath(__file__))
inner_dir = os.path.join(current_dir, 'Hunyuan3D-2-main')
if os.path.exists(inner_dir) and inner_dir not in sys.path:
    # Se houver a pasta extraída do .zip do github, adicione ela!
    sys.path.insert(0, inner_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = logging.getLogger("world_to_mesh.scene_mesh")


class SceneMeshGenerator:
    """
    Gerador de meshes 3D otimizado para cenários amplos.

    Adapta o pipeline Hunyuan3D-2 para processar mapas e paisagens
    gerados pelo HunyuanWorld, em vez de objetos isolados.

    Diferenças do pipeline padrão:
    - scene_mode=True: Pula background removal (cenários SÃO o fundo)
    - Resolução de octree maior para capturar detalhes de terreno
    - Mais inference steps para melhor qualidade em cenas complexas
    - Export direto como .glb

    Exemplo:
        >>> gen = SceneMeshGenerator(model_path="tencent/Hunyuan3D-2")
        >>> result = gen.generate(
        ...     image_path="/tmp/session/world_image.png",
        ...     output_path="/tmp/session/world_mesh.glb",
        ...     scene_mode=True
        ... )
        >>> print(result["glb_path"])
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2",
        device: str = "cuda",
        enable_texture: bool = True,
        low_vram: bool = False,
    ):
        """
        Inicializa o gerador de meshes.

        NOTA: A inicialização carrega os modelos na GPU. Em ambientes
        de GPU alugada, mantenha uma instância persistente para evitar
        recarregar os modelos a cada chamada.

        Args:
            model_path: Path do modelo no HuggingFace Hub
            device: Device para inferência ('cuda' ou 'cpu')
            enable_texture: Habilitar geração de texturas. Requer mais
                           VRAM (~16GB total vs ~6GB só shape)
            low_vram: Modo de baixa VRAM (offload para CPU quando possível)
        """
        self.model_path = model_path
        self.device = device
        self.enable_texture = enable_texture
        self.low_vram = low_vram

        # Modelos são carregados sob demanda (lazy loading)
        # para economizar memória quando não usados
        self._pipeline_shape = None
        self._pipeline_tex = None
        self._rembg = None

    def _load_models(self):
        """
        Carrega os modelos na GPU (lazy loading).

        Chamado automaticamente na primeira geração.
        Em GPU alugada, isso pode levar 30-60s na primeira vez
        (download do modelo do HuggingFace Hub + carregamento).
        """
        import torch

        if self._pipeline_shape is not None:
            return  # Já carregado

        logger.info(f"🔄 Carregando modelos Hunyuan3D-2 ({self.model_path})...")
        start = time.time()

        try:
            from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            import os
            import torch
            
            subfolder = "hunyuan3d-dit-v2-1"
            shape_subfolder = None
            for root, dirs, files in os.walk(self.model_path):
                for d in dirs:
                    if "hunyuan3d-dit" in d:
                        shape_subfolder = os.path.relpath(os.path.join(root, d), self.model_path)
                        # Se achar exatamente o v2-0 base, prioridade máxima
                        if d == "hunyuan3d-dit-v2-0":
                            break
                if shape_subfolder:
                    break
                    
            if shape_subfolder:
                subfolder = shape_subfolder
            else:
                logger.error(f"❌ NENHUM PESO 'hunyuan3d-dit' ENCONTRADO EM {self.model_path}")
                import sys
                sys.exit(1)
                
            logger.info(f"📂 Usando shape subfolder: {subfolder}")
            self._pipeline_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder=subfolder
            )
            logger.info(f"  ✅ Shape pipeline carregado ({time.time() - start:.1f}s)")
        except Exception as e:
            logger.error(f"  ❌ Falha ao carregar shape pipeline: {e}")
            raise

        if self.enable_texture:
            try:
                try:
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline
                except ImportError:
                    from hy3dpaint import Hunyuan3DPaintPipeline
                    
                tex_subfolder = "hunyuan3d-paint-v2-0-turbo"
                tex_found = None
                for root, dirs, files in os.walk(self.model_path):
                    for d in dirs:
                        if "hunyuan3d-paint" in d:
                            tex_found = os.path.relpath(os.path.join(root, d), self.model_path)
                            # prefer turbo if multiple exist
                            if "turbo" in d:
                                break
                    if tex_found and "turbo" in tex_found:
                        break
                        
                if tex_found:
                    tex_subfolder = tex_found
                else:
                    logger.error(f"❌ NENHUM PESO 'hunyuan3d-paint' ENCONTRADO EM {self.model_path}")
                    import sys
                    sys.exit(1)

                logger.info(f"📂 Usando tex subfolder: {tex_subfolder}")
                self._pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
                    self.model_path,
                    subfolder=tex_subfolder
                )
                logger.info(f"  ✅ Texture pipeline carregado ({time.time() - start:.1f}s)")
            except Exception as e:
                logger.warning(f"  ⚠️ Texture pipeline não disponível: {e}")
                self._pipeline_tex = None

        try:
            from hy3dgen.rembg import BackgroundRemover
            self._rembg = BackgroundRemover()
        except Exception as e:
            logger.warning(f"  ⚠️ Background remover não disponível: {e}")
            self._rembg = None

        logger.info(f"🔄 Modelos carregados em {time.time() - start:.1f}s")

    def generate(
        self,
        image_path: str,
        output_path: str,
        scene_mode: bool = True,
        octree_resolution: int = 256,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.5,
        seed: int = 42,
        apply_texture: bool = True,
        max_faces: int = 80000,
    ) -> dict:
        """
        Gera mesh 3D a partir de uma imagem.

        Para cenários (scene_mode=True):
        - Pula background removal (o cenário É o conteúdo)
        - Usa resolução de octree maior (256 vs 128 padrão)
        - Mais inference steps para melhor qualidade
        - Mais faces para capturar detalhes de terreno

        Para objetos isolados (scene_mode=False):
        - Aplica background removal normal
        - Parâmetros padrão do Hunyuan3D-2

        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho de saída do arquivo .glb
            scene_mode: True para cenários amplos, False para objetos
            octree_resolution: Resolução da octree (128, 256, ou 384)
            num_inference_steps: Número de steps de inferência
            guidance_scale: Escala de guidance (maior = mais fiel à imagem)
            seed: Seed para reprodutibilidade
            apply_texture: Aplicar texturas ao mesh
            max_faces: Número máximo de faces no mesh final

        Returns:
            dict com:
                - "success": bool
                - "glb_path": str (caminho do .glb gerado)
                - "vertices": int (número de vértices)
                - "faces": int (número de faces)
                - "time_seconds": float (tempo total de geração)
                - "error": str (se success=False)
        """
        import torch
        from PIL import Image

        result = {
            "success": False,
            "glb_path": "",
            "vertices": 0,
            "faces": 0,
            "time_seconds": 0.0,
            "error": "",
        }

        start_time = time.time()

        # ── Validar entrada ──────────────────────────────────────
        img_path = Path(image_path)
        if not img_path.exists():
            result["error"] = f"Imagem não encontrada: {image_path}"
            logger.error(result["error"])
            return result

        # ── Carregar modelos (lazy) ──────────────────────────────
        try:
            self._load_models()
        except Exception as e:
            result["error"] = f"Falha ao carregar modelos: {e}"
            return result

        # ── Carregar e preparar imagem ───────────────────────────
        try:
            image = Image.open(str(img_path)).convert("RGBA")
            logger.info(f"📷 Imagem carregada: {img_path.name} ({image.size[0]}x{image.size[1]})")
        except Exception as e:
            result["error"] = f"Falha ao carregar imagem: {e}"
            logger.error(result["error"])
            return result

        # ── Background Removal ───────────────────────────────────
        # PULO DO GATO: Em scene_mode, NÃO removemos o background.
        # Para mapas/paisagens, o "fundo" é parte essencial do conteúdo.
        # Remover o background destruiria as bordas do terreno e o céu.
        if scene_mode:
            logger.info("🌍 Modo CENÁRIO: Background removal DESABILITADO")
            # Converter para RGBA sem remover fundo
            # A imagem já está em RGBA
        else:
            if self._rembg is not None:
                logger.info("🎯 Modo OBJETO: Aplicando background removal")
                image = self._rembg(image)
            elif image.mode == "RGB":
                logger.warning("⚠️ Background remover não disponível, usando imagem original")
                image = image.convert("RGBA")

        # ── Geração do Shape ─────────────────────────────────────
        logger.info(
            f"🔨 Gerando mesh 3D (octree={octree_resolution}, "
            f"steps={num_inference_steps}, guidance={guidance_scale})..."
        )

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            mesh = self._pipeline_shape(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                generator=generator,
            )[0]

            logger.info("  ✅ Mesh gerado com sucesso")
        except Exception as e:
            result["error"] = f"Falha na geração de shape: {e}"
            logger.error(result["error"])
            return result

        # ── Pós-processamento ────────────────────────────────────
        try:
            try:
                from hy3dgen.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer
            except ImportError:
                from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer

            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=max_faces)
            logger.info(f"  ✅ Pós-processamento: max {max_faces} faces")
        except ImportError:
            logger.warning("  ⚠️ Pós-processadores não disponíveis, usando mesh cru")
        except Exception as e:
            logger.warning(f"  ⚠️ Erro no pós-processamento: {e}")

        # ── Texturização ─────────────────────────────────────────
        if apply_texture and self._pipeline_tex is not None:
            try:
                logger.info("🎨 Aplicando texturas...")
                mesh = self._pipeline_tex(mesh, image=image)
                logger.info("  ✅ Texturas aplicadas")
            except Exception as e:
                logger.warning(f"  ⚠️ Falha na texturização (mesh será sem textura): {e}")

        # ── Export como .glb ─────────────────────────────────────
        try:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Garantir extensão .glb
            if out_path.suffix.lower() != ".glb":
                out_path = out_path.with_suffix(".glb")

            mesh.export(str(out_path))

            # Coletar métricas do mesh
            try:
                result["vertices"] = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
                result["faces"] = len(mesh.faces) if hasattr(mesh, "faces") else 0
            except Exception:
                pass

            result["success"] = True
            result["glb_path"] = str(out_path)
            result["time_seconds"] = round(time.time() - start_time, 2)

            logger.info(
                f"✅ Mesh exportado: {out_path.name} "
                f"({result['vertices']} verts, {result['faces']} faces, "
                f"{result['time_seconds']}s)"
            )
        except Exception as e:
            result["error"] = f"Falha ao exportar .glb: {e}"
            logger.error(result["error"])

        # ── Liberar VRAM ─────────────────────────────────────────
        if self.low_vram:
            try:
                torch.cuda.empty_cache()
                logger.info("  🧹 VRAM liberada (low_vram mode)")
            except Exception:
                pass

        return result

    def unload_models(self):
        """
        Descarrega modelos da GPU para liberar VRAM.

        Útil em GPU alugada quando você quer alternar entre
        HunyuanWorld e Hunyuan3D-2 na mesma GPU.
        """
        import torch

        self._pipeline_shape = None
        self._pipeline_tex = None
        self._rembg = None

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("🧹 Modelos descarregados, VRAM liberada")


# ── CLI ────────────────────────────────────────────────────────────────
# Uso direto via terminal:
#   python scene_mesh_generator.py --input image.png --output mesh.glb --scene_mode

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scene Mesh Generator – Hunyuan3D-2 para cenários amplos"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Caminho da imagem de entrada",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Caminho de saída do arquivo .glb",
    )
    parser.add_argument(
        "--scene_mode", action="store_true", default=True,
        help="Modo cenário: desabilita background removal (padrão: True)",
    )
    parser.add_argument(
        "--object_mode", action="store_true", default=False,
        help="Modo objeto: habilita background removal",
    )
    parser.add_argument(
        "--model_path", type=str, default="tencent/Hunyuan3D-2",
        help="Path do modelo no HuggingFace Hub",
    )
    parser.add_argument(
        "--octree_resolution", type=int, default=256,
        help="Resolução da octree (128/256/384). Maior = mais detalhes",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=30,
        help="Número de steps de inferência (mais = melhor qualidade)",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.5,
        help="Escala de guidance (maior = mais fiel à imagem)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reprodutibilidade",
    )
    parser.add_argument(
        "--no_texture", action="store_true", default=False,
        help="Gerar mesh sem texturas (mais rápido, menos VRAM)",
    )
    parser.add_argument(
        "--max_faces", type=int, default=80000,
        help="Número máximo de faces no mesh final",
    )
    parser.add_argument(
        "--low_vram", action="store_true", default=False,
        help="Modo de baixa VRAM (libera memória agressivamente)",
    )
    parser.add_argument(
        "--normal_map", type=str, default=None,
        help="Path do normal map HD (gerado pelo StableNormal-turbo)",
    )
    args = parser.parse_args()

    # Determinar modo
    scene_mode = not args.object_mode

    print("=" * 60)
    print("  SCENE MESH GENERATOR")
    print("=" * 60)
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Mode:       {'CENÁRIO' if scene_mode else 'OBJETO'}")
    print(f"  Octree:     {args.octree_resolution}")
    print(f"  Steps:      {args.num_inference_steps}")
    print(f"  Guidance:   {args.guidance_scale}")
    print(f"  Texture:    {'Sim' if not args.no_texture else 'Não'}")
    print(f"  Max Faces:  {args.max_faces}")
    print("=" * 60)

    generator = SceneMeshGenerator(
        model_path=args.model_path,
        enable_texture=not args.no_texture,
        low_vram=args.low_vram,
    )

    result = generator.generate(
        image_path=args.input,
        output_path=args.output,
        scene_mode=scene_mode,
        octree_resolution=args.octree_resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        apply_texture=not args.no_texture,
        max_faces=args.max_faces,
    )

    if result["success"]:
        print(f"\n✅ SUCESSO!")
        print(f"   Arquivo: {result['glb_path']}")
        print(f"   Vértices: {result['vertices']}")
        print(f"   Faces: {result['faces']}")
        print(f"   Tempo: {result['time_seconds']}s")
    else:
        print(f"\n❌ FALHA: {result['error']}")
        sys.exit(1)
