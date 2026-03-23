"""
AI WORLD ENGINE — Teste Trellis 2 (Imagem → GLB) via Modal A100
Instala o Trellis via git clone + setup dentro do container.
Roda: modal run tests/modal_test_trellis.py
"""
import modal
import io
import os

# Imagem com Trellis 2 instalado via clone completo
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget", "build-essential")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "Pillow>=10.0.0",
        "numpy<2.0.0",
        "trimesh>=4.0.0",
        "huggingface_hub>=0.20.0",
        "safetensors>=0.4.0",
        "accelerate>=0.28.0",
        "transformers>=4.40.0",
        "scipy",
        "einops",
        "rembg",
        "onnxruntime",
    )
    .run_commands(
        # Clona Trellis com submodules
        "cd /opt && git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git",
        # Instala dependências do Trellis
        "cd /opt/TRELLIS && pip install -e .",
    )
)

app = modal.App("awe-trellis-v2", image=gpu_image)


@app.function(gpu="A100", timeout=1800)
def test_trellis(img_bytes: bytes):
    """Trellis: Imagem → GLB"""
    import sys
    sys.path.insert(0, "/opt/TRELLIS")

    import torch
    from PIL import Image

    print("=" * 50)
    print("ETAPA 3 — TRELLIS (Imagem → GLB)")
    print("=" * 50)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Imagem: {img.size}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

    # Importar Trellis
    print("Importando Trellis...")
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        print("✅ TrellisImageTo3DPipeline importado")
    except ImportError:
        # Lista o que existe
        import subprocess
        result = subprocess.run(["find", "/opt/TRELLIS", "-name", "*.py", "-path", "*/pipelines/*"],
                              capture_output=True, text=True)
        print(f"Arquivos de pipeline encontrados:\n{result.stdout}")

        # Tenta importar do path direto
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
            print("✅ Import alternativo OK")
        except Exception as e2:
            print(f"❌ Import falhou: {e2}")
            # Última tentativa: ver o que tem no trellis
            result2 = subprocess.run(["python", "-c", "import trellis; print(dir(trellis))"],
                                    capture_output=True, text=True)
            print(f"trellis dir: {result2.stdout}")
            print(f"trellis err: {result2.stderr}")
            return None

    print("Carregando TRELLIS-image-large...")
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    print("✅ Pipeline carregada")

    print("Gerando geometria 3D...")
    with torch.no_grad():
        outputs = pipe.run(
            img,
            seed=777,
            sparse_structure_sampler_params={"steps": 12},
            slat_sampler_params={"steps": 12},
        )

    print("Exportando GLB...")
    # Trellis pode retornar GLB de formas diferentes
    try:
        glb_data = outputs["mesh"][0].export_glb()
        if isinstance(glb_data, bytes):
            glb_bytes = glb_data
        else:
            buf = io.BytesIO()
            glb_data.save(buf)
            glb_bytes = buf.getvalue()
    except Exception as e:
        print(f"Export tentativa 1 falhou: {e}")
        try:
            mesh = outputs["mesh"][0]
            glb_bytes = mesh.export(file_type="glb")
        except Exception as e2:
            print(f"Export tentativa 2 falhou: {e2}")
            # Tenta trimesh
            import trimesh
            mesh_data = outputs["mesh"][0]
            if hasattr(mesh_data, 'vertices') and hasattr(mesh_data, 'faces'):
                tm = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces)
                glb_bytes = tm.export(file_type="glb")
            else:
                print(f"Tipo do mesh: {type(mesh_data)}")
                print(f"Atributos: {dir(mesh_data)}")
                return None

    print(f"✅ GLB gerado: {len(glb_bytes) // 1024} KB")

    del pipe, outputs
    torch.cuda.empty_cache()

    return glb_bytes


@app.local_entrypoint()
def main():
    print("\n🚀 AI WORLD ENGINE — TRELLIS (Imagem → GLB)")
    print("=" * 50)

    img_path = "outputs/output_flux.png"
    if not os.path.exists(img_path):
        print(f"❌ Imagem não encontrada: {img_path}")
        return

    with open(img_path, "rb") as f:
        img_bytes = f.read()
    print(f"Imagem: {len(img_bytes) // 1024} KB")

    print("\n>>> Gerando GLB na A100...")
    glb_bytes = test_trellis.remote(img_bytes)

    if glb_bytes:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/output_trellis.glb", "wb") as f:
            f.write(glb_bytes)
        size_kb = len(glb_bytes) / 1024
        print(f"\n📁 GLB salvo: outputs/output_trellis.glb ({size_kb:.0f} KB)")
        print("\n" + "=" * 50)
        print("🏆 PRIMEIRO GLB REAL DO PROJETO!")
        print("   Visualize: https://gltf-viewer.donmccurdy.com/")
        print("=" * 50)
    else:
        print("❌ Trellis falhou.")
