import setup

import torch
import time
from diffusers import CogVideoXPipeline

# Set experiment parameters. CogVideo 2b with simple config
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

torch.cuda.reset_peak_memory_stats()

# See how long it takes to do the creation
start = time.time()

output = pipe(
    prompt="A slow pan across a mountain landscape at sunrise",
    num_frames=49,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
)

elapsed = time.time() - start

# Other stats
peak_vram = torch.cuda.max_memory_allocated() / 1e9

print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Peak VRAM: {peak_vram:.2f} GB")

# Export video
from diffusers.utils import export_to_video
export_to_video(output.frames[0], "cogvideo-2b-test1.mp4", fps=8)

