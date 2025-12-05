"""
Config template generator for multi-architecture LoRA training.
Supports Z-Image, FLUX, Wan, and Qwen models.
Borrowed from the training_config_gui structure, optimized for fast training.
"""

import yaml
import os
from datetime import datetime


# Architecture-specific configuration defaults
ARCHITECTURE_CONFIGS = {
    "Z-Image Turbo": {
        "default_path": "Tongyi-MAI/Z-Image-Turbo",
        "arch": "zimage",
        "is_flux": False,
        "train_text_encoder": False,
        "noise_scheduler": "flowmatch",
        "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        "sample_guidance_scale": 1.0,
        "sample_steps": 8,
    },
    "FLUX.1-dev": {
        "default_path": "black-forest-labs/FLUX.1-dev",
        "arch": None,
        "is_flux": True,
        "train_text_encoder": False,
        "noise_scheduler": "flowmatch",
        "assistant_lora_path": None,
        "sample_guidance_scale": 3.5,
        "sample_steps": 20,
    },
    "Wan 2.2 High": {
        "default_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        "arch": "wan22_14b",
        "is_flux": False,
        "train_text_encoder": False,
        "noise_scheduler": "flowmatch",
        "assistant_lora_path": None,
        "sample_guidance_scale": 3.5,
        "sample_steps": 25,
        "wan_stage": "high",
        # Wan requires 4-bit quant with ARA for 24GB cards
        "qtype": "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
        "quantize_te": True,
        "qtype_te": "qfloat8",
        "cache_text_embeddings": True,
    },
    "Wan 2.2 Low": {
        "default_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        "arch": "wan22_14b",
        "is_flux": False,
        "train_text_encoder": False,
        "noise_scheduler": "flowmatch",
        "assistant_lora_path": None,
        "sample_guidance_scale": 3.5,
        "sample_steps": 25,
        "wan_stage": "low",
        # Wan requires 4-bit quant with ARA for 24GB cards
        "qtype": "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
        "quantize_te": True,
        "qtype_te": "qfloat8",
        "cache_text_embeddings": True,
    },
    "Wan 2.2 Combo": {
        "default_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        "arch": "wan22_14b",
        "is_flux": False,
        "train_text_encoder": False,
        "noise_scheduler": "flowmatch",
        "assistant_lora_path": None,
        "sample_guidance_scale": 3.5,
        "sample_steps": 25,
        "wan_stage": "combo",  # Trains both high and low noise stages
        # Wan requires 4-bit quant with ARA for 24GB cards
        "qtype": "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
        "quantize_te": True,
        "qtype_te": "qfloat8",
        "cache_text_embeddings": True,
    },
}


def _build_model_config(arch_config, model_path, quantize, low_vram, layer_offloading):
    """Build architecture-specific model config."""
    model_config = {
        'name_or_path': model_path,
        'quantize': quantize,
        'low_vram': low_vram,
        'layer_offloading': layer_offloading,
    }

    # Add architecture-specific fields
    if arch_config['arch']:
        model_config['arch'] = arch_config['arch']
    if arch_config['is_flux']:
        model_config['is_flux'] = True
    if arch_config['assistant_lora_path']:
        model_config['assistant_lora_path'] = arch_config['assistant_lora_path']
    # Wan-specific settings
    if arch_config.get('wan_stage'):
        stage = arch_config['wan_stage']
        model_config['model_kwargs'] = {
            'train_high_noise': stage in ['high', 'combo'],
            'train_low_noise': stage in ['low', 'combo'],
        }
    if arch_config.get('qtype'):
        model_config['qtype'] = arch_config['qtype']
    if arch_config.get('quantize_te'):
        model_config['quantize_te'] = arch_config['quantize_te']
    if arch_config.get('qtype_te'):
        model_config['qtype_te'] = arch_config['qtype_te']

    return model_config


def generate_training_config(
    name: str,
    image_folder: str,
    output_folder: str,
    architecture: str,
    model_path: str,
    steps: int = 100,
    learning_rate: float = 5e-4,
    lora_rank: int = 16,
    resolution: int = 1024,
    low_vram: bool = False,
    quantize: bool = False,
    layer_offloading: bool = False,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """
    Generate a training config dict for multi-architecture LoRA training.

    Supports Z-Image, FLUX, Wan, and Qwen models with architecture-specific settings.

    Args:
        name: Unique name for this training run
        image_folder: Path to folder containing the training image(s) and caption(s)
        output_folder: Where to save the trained LoRA
        architecture: Model architecture (Z-Image Turbo, FLUX.1-dev, Wan 2.2 High/Low, Qwen Image/Edit)
        model_path: HuggingFace path or local path to model
        steps: Number of training steps (default 100 for fast training)
        learning_rate: Learning rate (default 5e-4, higher than normal for fast overfitting)
        lora_rank: LoRA rank/dimension (default 16)
        resolution: Training resolution (default 1024)
        low_vram: Enable low VRAM mode (quantizes model on CPU, enables offloading)
        quantize: Enable model quantization for reduced VRAM usage
        layer_offloading: Enable layer offloading (moves transformer blocks between CPU/GPU)
        gradient_accumulation_steps: Number of gradient accumulation steps (trades time for VRAM)

    Returns:
        Config dict ready to be saved as YAML
    """
    # Get architecture-specific config
    arch_config = ARCHITECTURE_CONFIGS[architecture]

    config = {
        'job': 'extension',
        'config': {
            'name': name,
            'process': [
                {
                    'type': 'sd_trainer',
                    'training_folder': output_folder,
                    'device': 'cuda:0',

                    # Network config - standard LoRA
                    'network': {
                        'type': 'lora',
                        'linear': lora_rank,
                        'linear_alpha': lora_rank,
                    },

                    # Save config - only save final checkpoint
                    'save': {
                        'dtype': 'float16',
                        'save_every': steps,  # Only save at the end
                        'max_step_saves_to_keep': 1,
                    },

                    # Dataset config - single image folder
                    'datasets': [
                        {
                            'folder_path': image_folder,
                            'caption_ext': 'txt',
                            'caption_dropout_rate': 0.0,  # Always use caption for single image
                            'shuffle_tokens': False,
                            'cache_latents_to_disk': False,  # Not worth it for single image
                            'resolution': [resolution],
                            'bucket_no_upscale': True,
                        }
                    ],

                    # Training config - optimized for fast overfitting
                    'train': {
                        'batch_size': 1,
                        'steps': steps,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'train_unet': True,
                        'train_text_encoder': arch_config['train_text_encoder'],
                        'gradient_checkpointing': True,
                        'noise_scheduler': arch_config['noise_scheduler'],
                        'optimizer': 'adamw8bit',
                        'lr': learning_rate,
                        'cache_text_embeddings': arch_config.get('cache_text_embeddings', False),
                        'dtype': 'bf16',
                    },

                    # Model config - architecture-specific
                    'model': _build_model_config(arch_config, model_path, quantize, low_vram, layer_offloading),

                }
            ]
        },
        'meta': {
            'name': '[name]',
            'version': '1.0',
        }
    }

    return config


def save_config(config: dict, config_path: str) -> None:
    """Save config dict to YAML file."""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_unique_name() -> str:
    """Generate a unique name for a training run."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f"realtime_lora_{timestamp}"
