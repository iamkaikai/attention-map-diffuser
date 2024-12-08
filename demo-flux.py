import torch
from diffusers import FluxPipeline
import os
import random

from utils import (
    attn_maps,
    cross_attn_init,
    init_pipeline,
    save_attention_maps,
    GmPLoRALoader
)

shoe_anatomy_sections = [
    "toe",
    "box",
    "upper",
    "vamp",
    "throat",
    "collar",
    "tongue",
    "heel",
    "counter",
    "insole",
    "lining",
    "midsole",
    "outsole",
    "backstay",
    "eyelet",
    "eyestay",
    "quarter",
    "cap",
    "topline",
    "waist",
    "tread"
]
shoe_anatomy_sections = [section.lower() for section in shoe_anatomy_sections]


##### 1. Init redefined modules #####
cross_attn_init()
#####################################

# First load base components
pipe = FluxPipeline.from_pretrained( 
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.float32,
)

# Move everything to same device/dtype before LoRA
pipe = pipe.to(torch.float32)
# pipe = pipe.to("cpu")


# Load LoRA weights after moving to device
# pipe.load_lora_weights(
#     "./shoes-flux-dreambooth-lora-TE-GmP/checkpoint-7200", 
# )

# Load LoRA weights and apply GmP transformation
loader = GmPLoRALoader(pipe)
pipe = loader.load_lora_with_gmp(
    lora_path="./shoes-flux-dreambooth-lora-TE-GmP/checkpoint-7200",
    gmp_path="./shoes-flux-dreambooth-lora-TE-GmP/checkpoint-7200/gmp_params.pt"
)

# Enable optimizations after everything is loaded
# pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

##### 2. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################
# TE - p1 - [1090, 666, 977]
# TE - p2 - [333, 666, 1023]
# Base - p2 - [333, 666, ]
# GmP - p1 - [333, 666, 444, 555, 111, 1090, 977]
for seed in [666, 444, 555, 111, 1090, 977]:
    prompts = [
        "A side view of athletic shoes. The blue mesh upper with the black tongue and vamp, while the white midsole and black outsole.",
        # "lifestyle shoes; beige mesh upper; beige mesh tongue; black synthetic heel counter; white and grey EVA midsole; black rubber outsole"
    ]

    for batch, prompt in enumerate(prompts):
        # Create a unique directory for each prompt
        output_dir = f'output/prompt_{batch}_{seed}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate image
        generator = torch.Generator().manual_seed(seed)

        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            num_inference_steps=20,
            max_sequence_length=128,
            generator=generator
        ).images[0]

        # Save the generated image in the prompt-specific directory
        image.save(os.path.join(output_dir, f'generated_image_{batch}_{seed}.png'))

        ##### 3. Process and Save attention map #####
        save_attention_maps(
            attn_maps=attn_maps, 
            tokenizer=pipe.tokenizer, 
            prompts=prompts, 
            generated_image=image,
            base_dir=output_dir,  # Use the prompt-specific directory
            unconditional=False,
            alpha=0.75,
            parts=shoe_anatomy_sections
        )

print("Processing complete. Check the output directories for results.")