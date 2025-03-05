# !pip install -U controlnet-aux
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxPipeline, FluxControlPipeline
from diffusers.utils import load_image
import os


from utils import (
    GmPLoRALoader,
)

def canny_diversity(checkpoints, image_name, prompt, w, h):
    for checkpoint_name, name in checkpoints:
        #clean loader, pipe
        loader = None
        pipe = None
        del loader, pipe

        pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.float32).to("cuda")
        if checkpoint_name:
            if "GmP" in checkpoint_name:
                loader = GmPLoRALoader(pipe)
                pipe = loader.load_lora_with_gmp(
                    lora_path=f"{checkpoint_name}/pytorch_lora_weights.safetensors",
                    scale=1.0
                )
            else:
                pipe.load_lora_weights(checkpoint_name)
        
        prompt = prompt
        control_image = load_image(f"shoes-input/{image_name}")

        processor = CannyDetector()
        control_image = processor(control_image).convert("RGB")

        #make dir
        os.makedirs(f"output_canny_{checkpoint_name}/{name}", exist_ok=True)

        for seed in range(4):
            image = pipe(
                prompt=prompt,
                control_image=control_image,
                width=w,
                height=h,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(seed),
            ).images[0]
            image.save(f"output_canny_{checkpoint_name}/{name}/seed_{seed}.png")

def image_diversity(prompts, checkpoints):
    for checkpoint_name, name in checkpoints:
        #clean loader, pipe
        loader = None
        pipe = None
        del loader, pipe

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32).to("cuda")
        if checkpoint_name:
            if "GmP" in checkpoint_name:
                loader = GmPLoRALoader(pipe)
                pipe = loader.load_lora_with_gmp(
                    lora_path=f"{checkpoint_name}/pytorch_lora_weights.safetensors",
                    scale=1.0
                )
            else:
                pipe.load_lora_weights(checkpoint_name)
        
        #make dir
        os.makedirs(f"output_image/{name}", exist_ok=True)
        
        # Add prompt index to filename
        for prompt_idx, prompt in enumerate(prompts):
            for seed in range(4):
                image = pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    generator=torch.Generator().manual_seed(seed),
                ).images[0]
                # Include prompt_idx in the filename
                image.save(f"output_image/{name}/prompt_{prompt_idx}_seed_{seed}.png")


image_name= 'Artboard 1 copy.png'

checkpoints = [
    ("", "Dev"),
    ("shoes-flux-dreambooth-lora/checkpoint-4800", "DiT"),
    ("shoes-flux-dreambooth-lora-TE/checkpoint-4800", "DiT_TE"),
    ("shoes-flux-dreambooth-lora-TE-GmP/checkpoint-4800", "DiT_TE_GMP_4800"),
    ("shoes-flux-dreambooth-lora-TE-GmP/checkpoint-7200", "DiT_TE_GMP_7200"),
]
style_prompts = [
    # Tagging Prompts
    "running shoes; red mesh upper; gray TPU overlays; white EVA midsole; black rubber outsole; neon green heel support",
    "hiking boots; olive green leather upper; black rubber midsole; black lugged rubber outsole; yellow fabric laces",
    "basketball shoes; white knit upper; black synthetic overlays; blue TPU heel counter; white foam midsole; gum rubber outsole",
    "sandals; orange synthetic straps; black EVA footbed; dark brown rubber outsole",
    "oxfords; burgundy leather upper; dark brown leather sole; light brown stitching",
    "trail shoes; navy blue mesh upper; gray synthetic overlays; black rubber outsole; white EVA foam midsole",
    "cross-training shoes; teal knit upper; black synthetic overlays; gray TPU heel counter; white foam midsole; black rubber outsole",
    "cowboy boots; tan leather upper; brown leather outsole; white stitching; dark brown stacked heel",
    "flip-flops; pink rubber straps; black EVA footbed; white rubber outsole",
    "snow boots; gray fabric upper; black rubber outsole; white faux fur lining; gray laces",
    
    # Contextual Prompts (30-40 tokens)
    "Side view of red running shoes with breathable mesh upper, gray TPU overlays, white cushioned midsole, black rubber outsole, and neon green heel support.",
    "Side view of olive green hiking boots with rugged leather upper, black rubber midsole, thick lugged outsole, and yellow laces for secure fit.",
    "Side view of white basketball shoes featuring breathable knit upper, black overlays, blue heel counter, responsive foam midsole, and durable gum outsole.",
    "Top view of orange sandals with synthetic straps, black EVA footbed, and sturdy dark brown rubber outsole, designed for casual wear.",
    "Side view of burgundy oxford shoes with smooth leather upper, classic lace-up design, leather sole, and refined stitching details.",
    "Side view of navy blue trail shoes with mesh upper, gray overlays, cushioned EVA midsole, and durable black rubber outsole.",
    "Side view of teal cross-training shoes with knit upper, black overlays, gray heel counter, and responsive white foam midsole with durable outsole.",
    "Side view of tan cowboy boots with smooth leather upper, dark brown stacked heel, and decorative white stitching for a classic western style.",
    "Top view of pink flip-flops with rubber straps, black EVA footbed, and sturdy white rubber outsole, perfect for casual wear.",
    "Side view of gray snow boots with fabric upper, black rubber outsole, white faux fur lining, and gray laces for added warmth."
]

canny_prompts = [
    # "Side view of an athletic shoe featuring a beige upper, bold blue mesh with vibrant pink accents, and the prominent HOKA logo.",
    "Side view of an athletic shoe featuring a white upper with black overlays, light blue midsole, yellow accents, and the prominent orange HOKA logo."
]
canny_file_name = [
    # "Hoka CAD 1.png",
    "Hoka CAD 2.png"
]
dimensions = [
    # (1024, 512),
    (1024, 512),
]

for prompt, file_name, dim in zip(canny_prompts, canny_file_name, dimensions):
    canny_diversity(checkpoints, file_name, prompt, dim[0], dim[1])
# image_diversity(style_prompts, checkpoints)
