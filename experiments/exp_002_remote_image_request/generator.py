"""
generator.py - GPU machine (server side)

Runs a FastAPI HTTP server that:
    1. Loads SDXL-Turbo on startup (stays loaded in memory)
    2. Accepts POST /generate requests with a text prompt
    3. Runs image generation
    4. Returns the image as a PNG file

Run this on the machine with the GPU:
    python generator.py

Then send requests to it from any machine using requester.py

This server will be reachable at:
    http://<local-ip>:8000

To find local IP on Windows: run 'ipconfig' and find IPv4 Address
To find local IP on Mac/Linux: run 'ifconfig' or 'ip addr'

Dependencies:
    fastapi uvicorn diffusers transformers accelerate torch pillow
"""
import io
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

# App Setup
app = FastAPI(title="Remote Image Generator", version="0.1.0")

# pipline is loaded at startup then reused for requests
# Expect ~20s for initialization
pipeline = None

# Request / response models
# -------------------------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_steps: int = 4  # SDXL-Turbo works well at 4 steps
    seed: int = -1
    width: int = 512
    height: int = 512

# Startup: load model
# -------------------
@app.on_event("startup")
async def load_model():
    """
    Load SDXL-Turbo when server starts
    SDXL-Turbo is a simple version of SDXL that generates decent images. Toy model
    Standard models use ~50 steps. SDXL-Turbo uses about 4
    Can generate images in ~20s on RTX 3060, 12GB V-RAM

    We use torch.float16 to reduce memory usage
    """
    global pipeline

    print("Loading SDXL-Turbo... (this takes ~20s for first time)")
    print("Subsequent requests will be fast.\n")

    start = time.time()

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    # Move to GPU if available. Use CPU if not. CPU is MUCH slower
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        pipeline = pipeline.to("cpu")
        print("No GPU found. Running on CPU will be SLOW")
    elapsed = time.time() - start
    print(f"Model readied in {elapsed:.1f}s\n")
    print("Server is ready. Waiting for requests...")
    print("=" * 50)

# Routes
# ------
@app.get('/')
def root():
    """
    Verify server is up
    """
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        }
    return {
        "status": "ready",
        "model": "sdxl-turbo",
        "gpu": gpu_info
    }

@app.post("/generate")
def generate_image(request: GenerateRequest):
    """
    Generate an image from a text prompt.

    Accepts:
        JSON body with prompt, optional settings
    Returns:
        PNG image as binary response
    Example request body:
        {
            "prompt": "a red fox in a snow forest, digital art",
            "num_steps": 4,
            "seed": 42
        }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    print(f"\nReceived request:")
    print(f"  Prompt:\t{request.prompt!r}")
    print(f"  Steps:\t{request.num_steps}")
    print(f"  Seed:\t{request.seed}")
    print(f"  Size:\t{request.width}x{request.height}")

    # convert seed into generator
    if request.seed == -1:
        generator = None
    else:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(request.seed)

    start = time.time()

    # SDXL-Turbo doesn't use guidance scale, as speed optimization
    # Some linters don't like globals, thar be dragons :P
    result = pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_steps,
        guidance_scale=0.0,
        width=request.width,
        height=request.height,
        generator=generator
    )

    elapsed = time.time() - start
    image = result.images[0]  # PIL Image object

    print(f"  Generated in {elapsed:.1f}s")

    # Convert the PIL image to PNG bytes so we can send it over HTTP.
    # We write to an in-memory buffer instead of saving to disk.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    png_bytes = buffer.read()

    print(f"  Image size: {len(png_bytes) / 1024:.1f} KB")
    print(f"  Sending response...")

    # Return the raw PNG bytes with Content-Type header
    # requester can save this as .png file
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Generation-Time": str(round(elapsed, 2)),
            "X-Prompt": request.prompt[:100]
        }
    )

# Entry point
# -----------

if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" makes the server reachable from other machines on the network
    # use "127.0.0.1" for local access
    uvicorn.run(
        app,
        # host="127.0.0.1",
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )