import base64
import io
import threading
import torch
import runpod
from PIL import Image
from diffusers import FluxPipeline

_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
_pipe = None
_lock = threading.Lock()

def get_pipe():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                _pipe = FluxPipeline.from_pretrained(_MODEL_ID, torch_dtype=torch.float16)
                _pipe.enable_model_cpu_offload()  # safer VRAM usage
    return _pipe

def handler(job):
    data = job.get("input", {})
    prompt = data["prompt"]
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 4))
    seed = int(data.get("seed", 42))

    pipe = get_pipe()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,   # FLUX schnell style
        width=width,
        height=height,
        generator=gen
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": img_b64}

runpod.serverless.start({"handler": handler})
