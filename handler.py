import os
import io
import base64
import threading
from pathlib import Path

import runpod
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PIPE = None
LOCK = threading.Lock()


def _prepare_cache_dirs() -> None:
    hf_home = Path(os.getenv("HF_HOME", "/runpod-volume/hf"))
    hub_cache = Path(os.getenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    tr_cache = Path(os.getenv("TRANSFORMERS_CACHE", str(hf_home / "transformers")))

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    tr_cache.mkdir(parents=True, exist_ok=True)


def get_pipe() -> FluxPipeline:
    global PIPE
    if PIPE is None:
        with LOCK:
            if PIPE is None:
                _prepare_cache_dirs()

                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise RuntimeError("HF_TOKEN is missing in endpoint environment variables.")

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                PIPE = FluxPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    token=hf_token,
                )

                if torch.cuda.is_available():
                    PIPE.to("cuda")
                else:
                    PIPE.to("cpu")
    return PIPE


def handler(job):
    data = job.get("input", {})

    prompt = data.get("prompt")
    if not prompt:
        return {"error": "Missing required field: input.prompt"}

    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 4))
    seed = int(data.get("seed", 42))
    negative_prompt = data.get("negative_prompt")

    pipe = get_pipe()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=2.5,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_base64": image_b64,
        "seed": seed,
        "width": width,
        "height": height,
        "model": MODEL_ID,
    }


runpod.serverless.start({"handler": handler})
