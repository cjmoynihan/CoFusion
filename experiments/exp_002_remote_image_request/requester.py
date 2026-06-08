"""
Sends a text prompt to generator machine
Receives generated image and saves locally

Usage:
  python3 requester.py --host 192.168.1.42
  # with prompt:
  python3 requester.py --host 192.168.1.42 --prompt "a cat on a skateboard"
  # Other args: steps, seed, filename, ping

Dependencies:
    requests pillow
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import requests
from PIL import Image
import io

def ping_server(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        response.raise_for_status()
        info = response.json()

        print(f"Server UP at {base_url}")
        print(f"  Model: {info.get('model', 'unknown')}")

        gpu = info.get("gpu", {})
        if gpu:
            print(f"  GPU:\t{gpu.get('gpu', 'unknown')}")
            print(f"  VRAM:\t{gpu.get('vram_used_gb')} / {gpu.get('vram_total_gb')} GB used")
        else:
            print("  GPU:\trunning on CPU")

        return True
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {base_url}")
        print("  Is the generator running? Is the IP address correct?")
        return False
    except requests.exceptions.Timeout:
        print(f"ERROR: Connection timed out to {base_url}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

def request_image(
        base_url: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 4,
        seed: int = -1,
        width: int = 512,
        height: int = 512
) -> bytes:
    """
    Send a generation request to server and get back PNG image (bytes)
    Args:
        base_url:           e.g. "http://192.168.1.42:8000"
        prompt:             description of what to generate
        negative_prompt:    what to avoid in the image
        num_steps:          inference steps (4 is good for SDXL-Turbo)
        seed:               -1 for random, or fixed integer for reproducibility
        width / height:     image dimensions (must be multiple of 8)

    Returns:
        Raw PNG bytes. Save to a file or open with PIL
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_steps": num_steps,
        "seed": seed,
        "width": width,
        "height": height
    }

    print(f"\nSending request to {base_url}/generate ...")
    print(f"  Prompt:\t{prompt!r}")
    print(f"  Steps:\t{num_steps}  |  Seed: {seed}")
    print(f"  Waiting for image", end="", flush=True)

    start = time.time()

    try:
        # Send request
        # Expect possibly a longer timeout than normal
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=300
        )

        elapsed = time.time() - start
        print(f" done ({elapsed:.1f}s")

        # Check for HTTP response (4xx, 5xx)
        if response.status_code != 200:
            error_detail = response.json().get("detail", response.text)
            raise RuntimeError(f"Server returned error {response.status_code}: {error_detail}")

        # We should get raw PNG bytes (Content-Type: image/png)
        return response.content

    except requests.exceptions.ConnectionError:
        print()
        raise RuntimeError(
            f"Lost connection to {base_url}. "
            "Is the generator still running?"
        )
    except requests.exceptions.Timeout:
        print()
        raise RuntimeError(
            "Request timed out after 300s. "
            "Generation is taking very long. Check the generator machine."
        )

def save_image(png_bytes: bytes, output_path: str) -> Path:
    """
    Save PNG bytes to a file and return the path.
    Also opens the image with PIL to verify it's valid before saving.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Verify the bytes are a valid image before saving
    image = Image.open(io.BytesIO(png_bytes))
    image.save(path, format="PNG")

    return path

# Command Line Interface
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Send image generation requests to a remote GPU machine."
    )
    parser.add_argument(
        "--host",
        required=True,
        help="IP address of the generator machine (e.g. 192.168.1.42)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port the generator is listening on (default: 8000)"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Image generation prompt"
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt. What to avoid in the image."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4). Range 1-8 for SDXL-Turbo."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Set seed for reproducibility. (random by default)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512)"
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Output filename (default: output_<timestamp>.png)"
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Just check if the server is reachable, then exist."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Ping mode
    if args.ping:
        success = ping_server(base_url)
        sys.exit(0 if success else 1)

    # Check connectivity
    print(f"Connecting to generator at {base_url} ...")
    if not ping_server(base_url):
        sys.exit(1)

    prompt = args.prompt
    while not prompt:
        print()
        prompt = input("Enter your prompt: ".strip())

    # Destination filename
    if args.filename:
        output_path = args.filename
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.png"

    # Request image...
    try:
        png_bytes = request_image(
            base_url=base_url,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_steps=args.steps,
            seed=args.seed,
            width=args.width,
            height=args.height
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # Save image
    saved_path = save_image(png_bytes, output_path)
    print(f"\nImage saved to: {saved_path.resolve()}")
    print(f"Size: {len(png_bytes) / 1024:.1f} KB")

    print("\nDone! Open file to view image.")

if __name__ == "__main__":
    main()