import os
import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline
import types
import math
from PIL import Image, ImageEnhance
import numpy as np

# Configuration
BASE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT = "A photo of a cat wearing a hat"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize global attention storage
attn_maps = {}

# Patched attention forward method
def flux_attention_forward_patched(
    self, hidden_states, encoder_hidden_states=None, attention_mask=None, 
    image_rotary_emb=None, height=None, timestep=None
):
    batch_size, sequence_length, _ = hidden_states.shape

    # Compute queries, keys, values
    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads

    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    # Apply rotary embeddings if provided
    if image_rotary_emb is not None:
        text_length = encoder_hidden_states.shape[2] if encoder_hidden_states is not None else 0
        total_length = query.shape[2]
        image_length = total_length - text_length

        cos, sin = image_rotary_emb
        cos = cos[:image_length, :].view(1, 1, image_length, head_dim).to(query.dtype)
        sin = sin[:image_length, :].view(1, 1, image_length, head_dim).to(query.dtype)

        text_query = query[:, :, :text_length, :]
        image_query = query[:, :, text_length:, :]
        text_key = key[:, :, :text_length, :]
        image_key = key[:, :, text_length:, :]

        image_query = (image_query * cos) + (rotate_half(image_query) * sin)
        image_key = (image_key * cos) + (rotate_half(image_key) * sin)

        query = torch.cat([text_query, image_query], dim=2)
        key = torch.cat([text_key, image_key], dim=2)

    # Handle cross-attention
    if encoder_hidden_states is not None:
        encoder_hidden_states_query = self.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key = self.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value = self.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query = encoder_hidden_states_query.view(
            batch_size, -1, self.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key = encoder_hidden_states_key.view(
            batch_size, -1, self.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value = encoder_hidden_states_value.view(
            batch_size, -1, self.heads, head_dim
        ).transpose(1, 2)

        query = torch.cat([encoder_hidden_states_query, query], dim=2)
        key = torch.cat([encoder_hidden_states_key, key], dim=2)
        value = torch.cat([encoder_hidden_states_value, value], dim=2)

    # Compute attention
    attention_probs = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    attention_probs = attention_probs.softmax(dim=-1).to(value.dtype)
    hidden_states = torch.matmul(attention_probs, value)

    # Store attention maps if cross-attention
    if encoder_hidden_states is not None:
        layer_name = "unknown_layer"
        for name, module in pipe.transformer.named_modules():
            if module is self:
                layer_name = name
                break

        image_length = query.shape[2] - encoder_hidden_states_query.shape[2]
        cross_attn = attention_probs[:, :, :image_length, image_length:]
        attn_maps[layer_name] = attn_maps.get(layer_name, [])
        attn_maps[layer_name].append(cross_attn.detach().cpu())

    # Reshape hidden_states
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

    # Return output(s) based on cross-attention
    if encoder_hidden_states is not None:
        enc_len = encoder_hidden_states.shape[1]
        context_attn_output = hidden_states[:, :enc_len]
        attn_output = hidden_states[:, enc_len:]
        return attn_output, context_attn_output
    else:
        return hidden_states

# Helper function to rotate half
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Load pipeline
pipe = FluxPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Patch attention layers
print("\nPatching attention layers...")
patched_count = 0
for name, module in pipe.transformer.named_modules():
    if any(attention_type in module.__class__.__name__ for attention_type in [
        "FluxAttention", 
        "Attention",
        "CrossAttention",
        "BasicTransformerBlock"
    ]):
        print(f"Found attention layer: {name} ({module.__class__.__name__})")
        module.forward = types.MethodType(flux_attention_forward_patched, module)
        patched_count += 1

print(f"Patched {patched_count} attention layers")

# Generate image
print("Generating image...")
images = pipe(
    prompt=PROMPT,
    guidance_scale=0.0,
    height=1024,
    width=1024,
    num_inference_steps=12,
    max_sequence_length=512,
).images

# Save generated image
images[0].save(os.path.join(OUTPUT_DIR, "generated_image.png"))
print(f"Generated image saved to {os.path.join(OUTPUT_DIR, 'generated_image.png')}")

# Function to overlay attention map on the image
def overlay_attention_map(image, attention_map, output_path, alpha=0.5):
    """
    Overlay the attention map on the generated image.
    
    Args:
        image (PIL.Image.Image): The generated image.
        attention_map (np.ndarray): The attention map, normalized to [0, 1].
        output_path (str): Path to save the overlaid image.
        alpha (float): Transparency of the attention map overlay.
    """
    # Normalize attention map
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    attention_map = (attention_map * 255).astype(np.uint8)

    # Resize attention map to match image dimensions
    attention_map = Image.fromarray(attention_map).resize(image.size, resample=Image.BILINEAR)

    # Convert attention map to RGBA (heatmap)
    attention_map = attention_map.convert("L")
    heatmap = ImageEnhance.Color(attention_map.convert("RGBA")).enhance(1.5)

    # Blend the heatmap with the image
    blended_image = Image.blend(image.convert("RGBA"), heatmap, alpha=alpha)

    # Save the overlayed image
    blended_image.save(output_path)
    print(f"Saved overlayed image to {output_path}")

# Overlay attention maps on the generated image
for layer_name, attention_maps in attn_maps.items():
    print(f"Overlaying attention map for layer: {layer_name}")
    avg_attention = torch.mean(torch.stack(attention_maps), dim=0).mean(dim=1)[0].numpy()  # Average over heads
    generated_image = Image.open(os.path.join(OUTPUT_DIR, "generated_image.png"))
    output_overlay_path = os.path.join(OUTPUT_DIR, f"{layer_name}_overlay.png")
    overlay_attention_map(generated_image, avg_attention, output_overlay_path, alpha=0.5)