"""
AI WORLD ENGINE — Trellis via HuggingFace Spaces (Gradio API)
Roda: python tests/test_trellis_api.py
"""
import os
import sys
import json

try:
    from gradio_client import Client, handle_file
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio_client"])
    from gradio_client import Client, handle_file

print("=" * 50)
print("ETAPA 3 — TRELLIS (via HuggingFace Space)")
print("=" * 50)

img_path = "outputs/output_flux.png"
if not os.path.exists(img_path):
    print(f"❌ Imagem: {img_path} não encontrada")
    sys.exit(1)

print(f"Imagem: {img_path}")

# Conecta ao Space — tenta microsoft/TRELLIS e alternativas
spaces = [
    "microsoft/TRELLIS",
    "theseanlavery/TRELLIS",
]

client = None
for space_name in spaces:
    try:
        print(f"\nConectando ao Space: {space_name}...")
        client = Client(space_name)
        print(f"✅ Conectado a {space_name}")

        # Lista endpoints disponíveis
        print("\nEndpoints disponíveis:")
        try:
            api_info = client.view_api(print_info=False, return_format="dict")
            for endpoint_name, endpoint_info in api_info.get("named_endpoints", {}).items():
                print(f"  {endpoint_name}")
            for endpoint_name, endpoint_info in api_info.get("unnamed_endpoints", {}).items():
                print(f"  {endpoint_name}")
        except Exception as e:
            print(f"  (não foi possível listar: {e})")
            client.view_api(print_info=True)
        break
    except Exception as e:
        print(f"  ❌ Falhou: {e}")
        continue

if client is None:
    print("❌ Nenhum Space disponível")
    sys.exit(1)

# Tenta os endpoints mais comuns
endpoints_to_try = [
    "/image_to_3d",
    "/generate",
    "/predict",
    "/run",
]

result = None
for endpoint in endpoints_to_try:
    try:
        print(f"\nTentando endpoint: {endpoint}")
        result = client.predict(
            image=handle_file(img_path),
            api_name=endpoint,
        )
        print(f"✅ Endpoint {endpoint} retornou!")
        break
    except Exception as e:
        print(f"  ❌ {endpoint}: {e}")
        continue

if result is None:
    # Tenta com número do endpoint
    for i in range(10):
        try:
            print(f"\nTentando endpoint /{i}")
            result = client.predict(
                handle_file(img_path),
                fn_index=i,
            )
            print(f"✅ fn_index={i} retornou!")
            break
        except Exception as e:
            err = str(e)
            if "too many" in err.lower() or "expected" in err.lower():
                print(f"  /{i}: precisa de mais argumentos: {err[:100]}")
            else:
                print(f"  /{i}: {err[:100]}")
            continue

if result is None:
    print("\n❌ Nenhum endpoint funcionou.")
    sys.exit(1)

# Processa resultado
print(f"\n📦 Resultado (tipo): {type(result)}")
print(f"📦 Resultado (valor): {result}")

os.makedirs("outputs", exist_ok=True)

def save_if_glb(item, name="output_trellis.glb"):
    """Salva o item se for um path para um GLB"""
    if isinstance(item, str):
        if os.path.exists(item):
            import shutil
            dest = f"outputs/{name}"
            shutil.copy(item, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"🏆 GLB salvo: {dest} ({size_kb:.0f} KB)")
            return True
        elif item.startswith("http"):
            print(f"  URL: {item}")
            import urllib.request
            dest = f"outputs/{name}"
            urllib.request.urlretrieve(item, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"🏆 GLB baixado: {dest} ({size_kb:.0f} KB)")
            return True
    return False

found = False
if isinstance(result, tuple) or isinstance(result, list):
    for i, item in enumerate(result):
        print(f"\n  Output[{i}]: tipo={type(item)} valor={str(item)[:200]}")
        if save_if_glb(item, f"output_trellis_{i}.glb"):
            found = True
elif isinstance(result, dict):
    for key, item in result.items():
        print(f"\n  Output[{key}]: tipo={type(item)} valor={str(item)[:200]}")
        if save_if_glb(item, f"output_trellis_{key}.glb"):
            found = True
else:
    save_if_glb(result)

print("\n" + "=" * 50)
if found or os.path.exists("outputs/output_trellis.glb"):
    print("🏆 PRIMEIRO GLB REAL DO PROJETO!")
    print("   Visualize: https://gltf-viewer.donmccurdy.com/")
else:
    print("O resultado está acima. Pode conter URLs ou paths para download.")
print("=" * 50)
