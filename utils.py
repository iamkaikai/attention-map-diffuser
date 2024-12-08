import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import torch.nn as nn
import types


from diffusers.models import Transformer2DModel
from diffusers.models.unets import UNet2DConditionModel
from diffusers.models.transformers import SD3Transformer2DModel, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers import FluxPipeline
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)
import matplotlib.pyplot as plt
from modules import *

def cross_attn_init():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    JointAttnProcessor2_0.__call__ = joint_attn_call2_0
    FluxAttnProcessor2_0.__call__ = flux_attn_call2_0


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = getattr(module.processor, "timestep", None)
            attn_map = getattr(module.processor, "attn_map", None)

            if timestep is not None and attn_map is not None:
                attn_maps[timestep] = attn_maps.get(timestep, dict())
                attn_maps[timestep][name] = attn_map.cpu() if detach else attn_map
            
            # Clean up
            if hasattr(module.processor, "attn_map"):
                del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline):
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def group_tokens_into_words(tokens):
    words = []
    current_word = []
    
    for token in tokens:
        # Remove special tokens
        if token in ['<|startoftext|>', '<|endoftext|>']:
            continue
        
        # Check if the token ends with '</w>'
        if token.endswith('</w>'):
            # Remove the '</w>' suffix
            clean_token = token.replace('</w>', '')
            current_word.append(clean_token)
            # Combine all parts to form the complete word
            word = ''.join(current_word)
            words.append(word)
            # Reset for the next word
            current_word = []
        else:
            # Continue building the current word
            clean_token = token.replace('</w>', '')
            current_word.append(clean_token)
    
    # Handle any remaining tokens that didn't end with '</w>'
    if current_word:
        word = ''.join(current_word)
        words.append(word)
    
    return words

def map_words_to_token_indices(tokens):
    """
    Maps each word to its corresponding token indices.
    
    Args:
        tokens (List[str]): List of tokens from the tokenizer.
        
    Returns:
        List[List[int]]: A list where each sublist contains the token indices for a word.
    """
    word_to_token_indices = []
    current_indices = []
    
    for idx, token in enumerate(tokens):
        if token in ['<|startoftext|>', '<|endoftext|>']:
            continue
        
        current_indices.append(idx)
        
        if token.endswith('</w>'):
            word_to_token_indices.append(current_indices)
            current_indices = []
    
    # Handle any remaining tokens that didn't end with '</w>'
    if current_indices:
        word_to_token_indices.append(current_indices)
    
    return word_to_token_indices



def save_attention_maps(attn_maps, tokenizer, prompts, generated_image, base_dir='attn_maps', unconditional=True, alpha=0.5, parts=None):
    """
    Save the attention maps overlaid on the generated image.
    Args:
        attn_maps: Dictionary containing attention maps
        tokenizer: Tokenizer used for the prompts
        prompts: List of input prompts
        generated_image: The output image to overlay attention maps on (PIL Image or tensor)
        base_dir: Directory to save the outputs
        unconditional: Whether to use unconditional generation
        alpha: Transparency of the attention map overlay (0.0 to 1.0)
    """
    # Convert generated_image to PIL if it's a tensor
    if torch.is_tensor(generated_image):
        if generated_image.dim() == 4:
            generated_image = generated_image.squeeze(0)
        generated_image = ToPILImage()(generated_image)
    
    # Tokenize the prompts
    tokenized = tokenizer(prompts)
    token_ids = tokenized['input_ids']
    all_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids]
    all_words = [group_tokens_into_words(tokens) for tokens in all_tokens]
    all_word_token_indices = [map_words_to_token_indices(tokens) for tokens in all_tokens]
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Initialize total attention map
    total_attn_map = None
    total_attn_map_number = 0

    # Aggregate attention maps across timesteps
    for timestep, layers in attn_maps.items():
        for layer, attn_map in layers.items():
            # Sum over heads and prepare dimensions
            attn_map = attn_map.sum(1).squeeze(1)  # Shape: [batch, tokens, height, width]
            attn_map = attn_map.permute(0, 3, 1, 2)  # Shape: [batch, width, tokens, height]

            # Separate conditional/unconditional maps if necessary
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            # Resize to match desired output resolution
            if total_attn_map is None:
                total_attn_map = torch.zeros_like(
                    F.interpolate(
                        attn_map,
                        size=(attn_map.shape[-2], attn_map.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                )
            
            resized_attn_map = F.interpolate(
                attn_map,
                size=total_attn_map.shape[-2:], 
                mode='bilinear',
                align_corners=False
            )
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

    # Normalize the total attention map
    total_attn_map /= total_attn_map_number

    # Process each image in the batch
    for batch_idx, (words, word_token_indices, token_attns) in enumerate(zip(all_words, all_word_token_indices, total_attn_map)):
        # Clear all existing plots at the start of each batch
        plt.close('all')
        
        batch_dir = os.path.join(base_dir, f'batch-{batch_idx}')
        os.makedirs(batch_dir, exist_ok=True)
        
        # Convert generated_image for this batch
        if isinstance(generated_image, list):
            current_image = generated_image[batch_idx]
        else:
            current_image = generated_image
        
        # Convert PIL image to numpy array for consistent sizing
        current_image_np = np.array(current_image)
    
        for word, token_indices in zip(words, word_token_indices):
            if parts is not None and word.lower() not in parts:
                continue

            # Aggregate attention maps for the word's tokens (e.g., by averaging)
            aggregated_attn = torch.zeros(token_attns.shape[-2:], dtype=token_attns.dtype)
            valid_token_count = 0
            for idx in token_indices:
                if idx < token_attns.shape[0]:
                    aggregated_attn += token_attns[idx]
                    valid_token_count += 1
            if valid_token_count > 0:
                aggregated_attn /= valid_token_count  # Average the attention maps

            # Convert attention map to numpy
            attention_map = aggregated_attn.cpu().numpy()
            
            # Normalize attention map
            attention_map -= attention_map.min()
            attention_map /= attention_map.max()
            
            # Resize attention map to match image dimensions
            attention_map_resized = F.interpolate(
                torch.tensor(attention_map)[None, None],
                size=(current_image_np.shape[0], current_image_np.shape[1]),  # (height, width)
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Optionally flip attention map if needed
            # attention_map_resized = np.flipud(attention_map_resized)
            
            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot the current batch's image
            ax.imshow(current_image_np)
            
            # Overlay attention map with transparency
            ax.imshow(attention_map_resized, cmap='jet', alpha=alpha)
            
            # Remove axes and padding
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Sanitize word for filename
            sanitized_word = word.replace('/', '_').replace('\\', '_')
            
            # Save and close
            os.makedirs(os.path.join(batch_dir, f'batch-{batch_idx}'), exist_ok=True)
            plt.savefig(
                os.path.join(batch_dir, f'batch-{batch_idx}/{sanitized_word}.png'),
                bbox_inches='tight', 
                pad_inches=0,
                dpi=72
            )
            plt.close(fig)        # Clear all existing plots at the start of each batch
        plt.close('all')


class GmPLayer(nn.Module):
    """Geometric Parametrization Layer for more stable Text Encoder fine-tuning"""
    def __init__(self, weight_matrix, bias=None):
        super().__init__()
        out_features, in_features = weight_matrix.shape
        device = weight_matrix.device  # Get the device from weight matrix
        
        # Initialize on same device
        self.theta = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.r = nn.Parameter(torch.empty(out_features, 1, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device))
        
        # Initialize values
        nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.r, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure same device
        x = x.to(self.theta.device)
        u = F.normalize(self.theta, p=2, dim=1)
        return F.linear(x, self.r * u) + self.bias


class GmPLoRALoader:
    def __init__(self, pipe):
        self.pipe = pipe
        
    def load_lora_with_gmp(self, lora_path, gmp_path=None):
        
        print(f"Loading LoRA weights from {lora_path}")
        self.pipe.load_lora_weights(lora_path)
        
        print("Applying GmP to LoRA layers...")
        for name, module in self.pipe.text_encoder.named_modules():
            if "lora_" in name and isinstance(module, nn.Linear):
                if "lora_A" in name or "lora_B" in name:
                    # Create GmP layer
                    gmp = GmPLayer(module.weight, bias=module.bias)
                    
                    # Replace forward method
                    def new_forward(self, x, gmp_layer=gmp):
                        return gmp_layer(x)
                    module.forward = types.MethodType(new_forward, module)
                    
                    # Store GmP parameters
                    module.gmp_theta = gmp.theta
                    module.gmp_r = gmp.r
                    module.gmp_bias = gmp.bias
                    
                    # Disable original LoRA gradients
                    module.weight.requires_grad_(False)
                    print(f"Applied GmP to {name}")
        
        # 3. Load GmP parameters if provided
        if gmp_path is None:
            gmp_path = os.path.join(lora_path, "gmp_parameters.pt")
        
        if os.path.exists(gmp_path):
            print(f"Loading GmP parameters from {gmp_path}")
            gmp_state = torch.load(gmp_path)
            for name, module in self.pipe.text_encoder.named_modules():
                if hasattr(module, 'gmp_theta'):
                    key = f"{name}.gmp"
                    if key in gmp_state:
                        module.gmp_theta.data = gmp_state[key]['theta']
                        module.gmp_r.data = gmp_state[key]['r']
                        module.gmp_bias.data = gmp_state[key]['bias']
                        print(f"Loaded GmP parameters for {name}")
        else:
            print(f"No GmP parameters found at {gmp_path}")
        
        return self.pipe
