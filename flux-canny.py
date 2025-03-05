# !pip install -U controlnet-aux
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "Side view of red hiking boots with a rugged black midsole, black rubber outsole, black laces, and a white logo on the toe box, designed for outdoor exploration"
control_image = load_image("shoes-input/Plan de travail 1.png")

processor = CannyDetector()
control_image = processor(control_image).convert("RGB")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=512,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")