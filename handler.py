import os
import io
import base64
import threading

import runpod
import torch
from diffusers import FluxPipeline


_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
_PIPE = None
_LOCK = threading.Lock()


def get_pipe():
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise RuntimeError("HF_TOKEN is missing in endpoint environment variables.")

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                _PIPE = FluxPipeline.from_pretrained(
                    _MODEL_ID,
                    torch_dtype=dtype,
                    token=hf_token,
                )

                if torch.cuda.is_available():
                    _PIPE.to("cuda")
                else:
                    _PIPE.to("cpu")
    return _PIPE


def handler(job):
    data = job.get("input", {})

    prompt = data.get("prompt")
    if not prompt:
        return {"error": "Missing required field: input.prompt"}

    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 4))
    seed = int(data.get("seed", 42))
    negative_prompt = data.get("negative_prompt", None)

    pipe = get_pipe()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    # FLUX.1-schnell typically uses low guidance (0.0 is common).
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
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
        "model": _MODEL_ID,
    }


runpod.serverless.start({"handler": handler})
