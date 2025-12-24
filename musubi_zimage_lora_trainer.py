"""
Musubi Tuner Z-Image LoRA Trainer Node for ComfyUI

Trains Z-Image LoRAs using kohya-ss/musubi-tuner.
Alternative to AI-Toolkit for Z-Image training.
"""

import os
import sys
import json
import glob
import hashlib
import tempfile
import shutil
import subprocess
import socket
import threading
import time
import signal
import queue
from datetime import datetime
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management

from .musubi_zimage_config_template import (
    generate_dataset_config,
    save_config,
    MUSUBI_ZIMAGE_VRAM_PRESETS,
)


# Global config for Musubi Z-Image trainer
_musubi_config = {}
_musubi_config_file = os.path.join(os.path.dirname(__file__), ".musubi_zimage_config.json")

# Global cache for trained LoRAs
_musubi_lora_cache = {}
_musubi_cache_file = os.path.join(os.path.dirname(__file__), ".musubi_zimage_lora_cache.json")


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division for positive integers."""
    if b <= 0:
        return a
    return (a + b - 1) // b


def _load_musubi_config():
    """Load Musubi config from disk."""
    global _musubi_config
    if os.path.exists(_musubi_config_file):
        try:
            with open(_musubi_config_file, 'r', encoding='utf-8') as f:
                _musubi_config = json.load(f)
        except:
            _musubi_config = {}


def _save_musubi_config():
    """Save Musubi config to disk."""
    try:
        with open(_musubi_config_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_config, f, indent=2)
    except:
        pass


def _load_musubi_cache():
    """Load Musubi LoRA cache from disk."""
    global _musubi_lora_cache
    if os.path.exists(_musubi_cache_file):
        try:
            with open(_musubi_cache_file, 'r', encoding='utf-8') as f:
                _musubi_lora_cache = json.load(f)
        except:
            _musubi_lora_cache = {}


def _save_musubi_cache():
    """Save Musubi LoRA cache to disk."""
    try:
        with open(_musubi_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_lora_cache, f)
    except:
        pass


def _compute_image_hash(images, captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=False):
    """Compute a hash of all images, captions, and training parameters."""
    hasher = hashlib.sha256()

    if use_folder_path:
        # For folder paths, hash the file paths and modification times
        for img_path in images:
            hasher.update(img_path.encode('utf-8'))
            if os.path.exists(img_path):
                hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))
    else:
        # For tensor inputs, hash the image data
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Include all captions in hash
    captions_str = "|".join(captions)
    params_str = f"musubi_zimage|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


def _get_venv_python_path(musubi_path):
    """Get the Python path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(musubi_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(musubi_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "python")


def _get_accelerate_path(musubi_path):
    """Get the accelerate path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(musubi_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(musubi_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "accelerate")


def _get_model_path(name, folder_type):
    """Get full path to a model file from ComfyUI folders.
    Returns the name as-is if it's already an absolute path that exists."""
    if not name:
        return ""
    # If it's already an absolute path that exists, use it
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Try to get from ComfyUI folder
    try:
        return folder_paths.get_full_path(folder_type, name)
    except:
        return name


# Filesystem-safe per-LoRA naming (folder + file base name)
_INVALID_FS_CHARS = set('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _validate_lora_name(output_name: str) -> str:
    """Validate output_name so it can be used as both a folder name and file base name."""
    if output_name is None:
        raise ValueError("output_name must be provided.")

    name = output_name.strip()
    if not name:
        raise ValueError("output_name must not be empty.")

    bad_chars = sorted({c for c in name if c in _INVALID_FS_CHARS})
    if bad_chars:
        raise ValueError(
            f"output_name contains invalid filesystem character(s): {''.join(bad_chars)}. "
            "Please use only letters/numbers/spaces/_-."
        )

    # Windows filesystem restrictions
    if sys.platform == "win32":
        if name.endswith(" ") or name.endswith("."):
            raise ValueError("output_name may not end with a space or dot on Windows.")
        base = name.split(".")[0].upper()
        if base in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"output_name '{name}' is a reserved Windows name.")

    return name


def _find_latest_state_dir(lora_folder: str) -> str:
    """Return the newest '*-state' directory inside lora_folder, or '' if none exist."""
    if not os.path.isdir(lora_folder):
        return ""

    candidates = []
    for entry in os.listdir(lora_folder):
        p = os.path.join(lora_folder, entry)
        if os.path.isdir(p) and entry.endswith("-state"):
            candidates.append(p)

    if not candidates:
        return ""

    return max(candidates, key=lambda p: os.path.getmtime(p))


def _reset_lora_run_folder(lora_folder: str, dataset_dir: str, cache_dir: str):
    """Delete dataset/, cache/, and any '*-state' directories within lora_folder."""
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    for entry in os.listdir(lora_folder):
        p = os.path.join(lora_folder, entry)
        if os.path.isdir(p) and entry.endswith("-state"):
            shutil.rmtree(p)


# Load config and cache on module import
_load_musubi_config()
_load_musubi_cache()


def _throw_if_comfyui_interrupted():
    """Raise ComfyUI's InterruptProcessingException if the user pressed Cancel."""
    comfy.model_management.throw_exception_if_processing_interrupted()


def _terminate_process_tree(proc: subprocess.Popen):
    """
    Terminate a subprocess and its children (multi-GPU training spawns additional python processes).

    We intentionally kill only the specific PID tree we started (never global python kills).
    """
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass

    if sys.platform == "win32":
        # Kill full process tree for the exact PID (covers torch.multiprocessing.spawn children).
        # /T = terminate child processes, /F = force.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    else:
        # On POSIX, if we start a new session, pid == pgid and we can kill the process group.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass


def _popen_with_process_group(*popen_args, **popen_kwargs) -> subprocess.Popen:
    """Start a subprocess in its own process group/session so we can terminate it reliably."""
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = popen_kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(*popen_args, **popen_kwargs)


def _run_subprocess_with_cancel(cmd, *, cwd, env, startupinfo, prefix: str):
    """
    Run a subprocess while:
    - streaming its output
    - frequently checking the ComfyUI cancel flag
    - killing the subprocess tree if cancelled
    """
    _throw_if_comfyui_interrupted()

    proc = _popen_with_process_group(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        startupinfo=startupinfo,
        env=env,
    )

    q: "queue.Queue[str]" = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                q.put(line)
        except Exception:
            # If the process is terminated (cancel), stdout can error/close; that's fine.
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        while True:
            _throw_if_comfyui_interrupted()

            # Drain any available output without blocking.
            drained_any = False
            while True:
                try:
                    line = q.get_nowait()
                except queue.Empty:
                    break

                drained_any = True
                line = line.rstrip()
                if line:
                    print(f"{prefix}{line}")

            ret = proc.poll()
            if ret is not None:
                # Final drain (best-effort)
                while True:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        break
                    line = line.rstrip()
                    if line:
                        print(f"{prefix}{line}")
                return ret

            # If no output, avoid busy-waiting but still react quickly to cancel.
            if not drained_any:
                time.sleep(0.1)

    except comfy.model_management.InterruptProcessingException:
        _terminate_process_tree(proc)
        raise
    finally:
        # Ensure we don't leave the subprocess running if an unexpected exception bubbles out.
        if proc.poll() is None:
            _terminate_process_tree(proc)

        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def _run_parallel_subprocesses_with_cancel(processes, *, cwd, startupinfo):
    """
    Run multiple subprocesses in parallel while:
    - streaming each process's stdout
    - frequently checking the ComfyUI cancel flag
    - killing ALL subprocess trees if cancelled or if any subprocess fails

    Args:
        processes: list of dicts, each with:
            - cmd: list[str]
            - env: dict[str, str]
            - prefix: str (printed before each output line)

    Returns:
        list[int]: return codes in the same order as `processes`.
    """
    _throw_if_comfyui_interrupted()

    procs: list[dict] = []

    def _start_one(p):
        proc = _popen_with_process_group(
            p["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            startupinfo=startupinfo,
            env=p["env"],
        )
        q: "queue.Queue[str]" = queue.Queue()

        def _reader():
            try:
                for line in proc.stdout:
                    q.put(line)
            except Exception:
                # If the process is terminated (cancel/fail), stdout can error/close; that's fine.
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return {"proc": proc, "q": q, "prefix": p.get("prefix", ""), "cmd": p["cmd"]}

    # Start all processes first (fail fast if any cannot be started).
    for p in processes:
        procs.append(_start_one(p))

    try:
        while True:
            _throw_if_comfyui_interrupted()

            # Drain any available output without blocking.
            drained_any = False
            for p in procs:
                q = p["q"]
                prefix = p["prefix"]
                while True:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    line = line.rstrip()
                    if line:
                        print(f"{prefix}{line}")

            # Poll processes and fail fast on first non-zero exit.
            all_done = True
            for p in procs:
                proc: subprocess.Popen = p["proc"]
                ret = proc.poll()
                if ret is None:
                    all_done = False
                    continue
                if ret != 0:
                    # Kill everything we started, then raise with context.
                    for other in procs:
                        _terminate_process_tree(other["proc"])
                    raise RuntimeError(f"Subprocess failed with code {ret}: {p['cmd']}")

            if all_done:
                # Final drain (best-effort)
                for p in procs:
                    q = p["q"]
                    prefix = p["prefix"]
                    while True:
                        try:
                            line = q.get_nowait()
                        except queue.Empty:
                            break
                        line = line.rstrip()
                        if line:
                            print(f"{prefix}{line}")

                return [int(p["proc"].returncode or 0) for p in procs]

            # If no output, avoid busy-waiting but still react quickly to cancel.
            if not drained_any:
                time.sleep(0.1)

    except comfy.model_management.InterruptProcessingException:
        for p in procs:
            _terminate_process_tree(p["proc"])
        raise
    finally:
        # Ensure we don't leave subprocesses running if an unexpected exception bubbles out.
        for p in procs:
            proc: subprocess.Popen = p["proc"]
            if proc.poll() is None:
                _terminate_process_tree(proc)
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

def _parse_device_indexes(device_indexes_str):
    """
    Parse comma-separated device indexes string.
    
    Args:
        device_indexes_str: Comma-separated GPU IDs (e.g., "0,1" or "0,1,2")
    
    Returns:
        tuple: (list of GPU IDs, number of GPUs)
    
    Raises:
        ValueError: If format is invalid
    """
    if not device_indexes_str or not device_indexes_str.strip():
        return [], 0
    
    try:
        # Split by comma and strip whitespace
        device_ids = [int(x.strip()) for x in device_indexes_str.split(',') if x.strip()]
        
        # Validate all are non-negative integers
        if any(d < 0 for d in device_ids):
            raise ValueError("Device indexes must be non-negative integers")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for d in device_ids:
            if d not in seen:
                seen.add(d)
                unique_ids.append(d)
        
        return unique_ids, len(unique_ids)
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid device_indexes format: '{device_indexes_str}'. Expected comma-separated integers (e.g., '0,1' or '0,1,2')")
        raise


def _get_free_port(default_port: int = 29500) -> int:
    """Return a free localhost TCP port, or default_port if allocation fails."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])
    except Exception:
        return int(default_port)


class MusubiZImageLoraTrainer:
    """
    Trains a Z-Image LoRA from one or more images using Musubi Tuner.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            musubi_fallback = 'C:\\musubi-tuner'
        else:
            musubi_fallback = '~/musubi-tuner'

        saved = _musubi_config.get('trainer_settings', {})
        default_vram_mode = saved.get('vram_mode', "Low (768px)")
        default_preset = MUSUBI_ZIMAGE_VRAM_PRESETS.get(default_vram_mode) or MUSUBI_ZIMAGE_VRAM_PRESETS.get("Low (768px)")


        # Get available models from ComfyUI folders
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        # Text encoders can be in clip or text_encoders folder
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        text_encoder_list = sorted(set(text_encoders + clip_models))

        # Get saved model selections (for default)
        saved_dit = saved.get('dit_model', '')
        saved_vae = saved.get('vae_model', '')
        saved_te = saved.get('text_encoder', '')

        # Build dropdown configs with saved defaults if available
        dit_config = {"tooltip": "Z-Image DiT model (transformer) from diffusion_models folder."}
        if saved_dit and saved_dit in diffusion_models:
            dit_config["default"] = saved_dit

        vae_config = {"tooltip": "Z-Image VAE model from vae folder."}
        if saved_vae and saved_vae in vae_models:
            vae_config["default"] = saved_vae

        te_config = {"tooltip": "Qwen3 text encoder from text_encoders or clip folder."}
        if saved_te and saved_te in text_encoder_list:
            te_config["default"] = saved_te

        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "musubi_path": ("STRING", {
                    "default": _musubi_config.get('musubi_path', musubi_fallback),
                    "tooltip": "Path to musubi-tuner installation."
                }),
                "dit_model": (diffusion_models, dit_config),
                "vae_model": (vae_models, vae_config),
                "text_encoder": (text_encoder_list, te_config),
                "caption": ("STRING", {
                    "default": saved.get('caption', "photo of subject"),
                    "multiline": True,
                    "tooltip": "Default caption for all images. Per-image caption inputs override this."
                }),
                "training_steps": ("INT", {
                    "default": saved.get('training_steps', 400),
                    "min": 10,
                    "max": 1000000,
                    "step": 10,
                    "tooltip": "Number of training steps. 400 is a good starting point."
                }),
                "learning_rate": ("FLOAT", {
                    "default": saved.get('learning_rate', 0.0002),
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "Learning rate. 0.0002 is recommended for Z-Image training."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 16),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 16 is recommended for Z-Image."
                }),
                "vram_mode": (["Max (1256px)", "Max (1256px) fp8", "Max (1256px) fp8 offload", "Medium (1024px)", "Medium (1024px) fp8", "Medium (1024px) fp8 offload", "Low (768px)", "Min (512px)"], {
                    "default": saved.get('vram_mode', "Low (768px)"),
                    "tooltip": "VRAM optimization preset. Low/Min always use fp8. Min adds pre-caching for lowest VRAM."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyLora"),
                    "tooltip": "Custom name for the output LoRA. Each output_name gets its own folder under musubi/output/."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside musubi_path. The musubi_path field is still required for locating training scripts."
                }),
                "enable_multi_gpu": ("BOOLEAN", {
                    "default": saved.get('enable_multi_gpu', False),
                    "tooltip": "Enable multi-GPU training. When enabled, uses device_indexes to select GPUs."
                }),
                "device_indexes": ("STRING", {
                    "default": saved.get('device_indexes', ""),
                    "tooltip": "Comma-separated GPU device IDs (e.g., '0,1' or '0,1,2'). Only used when enable_multi_gpu is True."
                }),
                "reuse_existing_caches": ("BOOLEAN", {
                    "default": saved.get('reuse_existing_caches', True),
                    "tooltip": "Reuse existing latent/text-encoder caches inside this LoRA's output folder. When enabled, caching will skip files that already exist."
                }),
                "save_training_state": ("BOOLEAN", {
                    "default": saved.get('save_training_state', False),
                    "tooltip": "Save training state (optimizer, scheduler, etc.). Required to resume training later."
                }),
                "continue_from_state": ("BOOLEAN", {
                    "default": saved.get('continue_from_state', False),
                    "tooltip": "Resume training from the newest saved state inside this LoRA's output folder."
                }),
                "reset_existing_run": ("BOOLEAN", {
                    "default": saved.get('reset_existing_run', False),
                    "tooltip": "DANGEROUS: Deletes this LoRA's dataset/cache/state folders to start fresh. Required if the LoRA folder already exists and you're NOT resuming."
                }),
                "checkpoint_every_n_steps": ("INT", {
                    "default": saved.get('checkpoint_every_n_steps', 0),
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Save a checkpoint every N steps. 0 disables step-based checkpointing. Note: On Windows multi-GPU, the node auto-scales this value to keep the same checkpoint cadence relative to training_steps."
                }),
                "checkpoint_every_n_epochs": ("INT", {
                    "default": saved.get('checkpoint_every_n_epochs', 0),
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Save a checkpoint every N epochs. 0 disables epoch-based checkpointing. Note: Musubi does not save on the final epoch (it always saves the final model at the end)."
                }),
                "warmup_steps": ("INT", {
                    "default": saved.get('warmup_steps', 0),
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Learning rate warmup steps. If > 0, the trainer will auto-use lr_scheduler=constant_with_warmup."
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": saved.get('gradient_checkpointing', bool(default_preset.get('gradient_checkpointing', False)) if default_preset else False),
                    "tooltip": "Enable gradient checkpointing (overrides VRAM preset). Recommended for Low/Min VRAM modes."
                }),
                "batch_size": ("INT", {
                    "default": saved.get('batch_size', int(default_preset.get('batch_size', 1)) if default_preset else 1),
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Training batch size (per process)."
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": saved.get('gradient_accumulation_steps', 1),
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Accumulate gradients for N steps before optimizer update."
                }),
                "enable_bucket": ("BOOLEAN", {
                    "default": saved.get('enable_bucket', True),
                    "tooltip": "Enable bucketed resizing (recommended)."
                }),
                "bucket_no_upscale": ("BOOLEAN", {
                    "default": saved.get('bucket_no_upscale', False),
                    "tooltip": "If enabled, buckets will not upscale smaller images."
                }),
                "num_repeats": ("INT", {
                    "default": saved.get('num_repeats', 10),
                    "min": 1,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "How many times each item is repeated per epoch (dataset_config num_repeats)."
                }),
                "enable_tensorboard": ("BOOLEAN", {
                    "default": saved.get('enable_tensorboard', False),
                    "tooltip": "Enable TensorBoard logging. Logs will be saved to a 'logs' folder inside your LoRA output folder. Run 'tensorboard --logdir <path>' to view."
                }),
                "tensorboard_port": ("INT", {
                    "default": saved.get('tensorboard_port', 6006),
                    "min": 1024,
                    "max": 65535,
                    "step": 1,
                    "tooltip": "Port for TensorBoard server (default: 6006). Use this when running 'tensorboard --logdir <path> --port <port>'."
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Training image (not needed if images_path is set)."}),
                "caption_1": ("STRING", {"forceInput": True, "tooltip": "Caption for image_1. Overrides default caption."}),
                "image_2": ("IMAGE", {"tooltip": "Training image."}),
                "caption_2": ("STRING", {"forceInput": True, "tooltip": "Caption for image_2. Overrides default caption."}),
                "image_3": ("IMAGE", {"tooltip": "Training image."}),
                "caption_3": ("STRING", {"forceInput": True, "tooltip": "Caption for image_3. Overrides default caption."}),
                "image_4": ("IMAGE", {"tooltip": "Training image."}),
                "caption_4": ("STRING", {"forceInput": True, "tooltip": "Caption for image_4. Overrides default caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained Z-Image LoRA file (ComfyUI format).",)
    FUNCTION = "train_zimage_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a Z-Image LoRA from images using Musubi Tuner. Lighter alternative to AI-Toolkit."

    def train_zimage_lora(
        self,
        inputcount,
        images_path,
        musubi_path,
        dit_model,
        vae_model,
        text_encoder,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        keep_lora=True,
        output_name="MyLora",
        custom_python_exe="",
        enable_multi_gpu=False,
        device_indexes="",
        reuse_existing_caches=True,
        save_training_state=False,
        continue_from_state=False,
        reset_existing_run=False,
        checkpoint_every_n_steps=0,
        checkpoint_every_n_epochs=0,
        warmup_steps=0,
        gradient_checkpointing=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        enable_bucket=True,
        enable_tensorboard=False,
        tensorboard_port=6006,
        bucket_no_upscale=False,
        num_repeats=10,
        image_1=None,
        **kwargs
    ):
        global _musubi_lora_cache

        # Expand paths
        musubi_path = os.path.expanduser(musubi_path.strip())

        # Get full paths from ComfyUI folders
        dit_path = _get_model_path(dit_model, "diffusion_models")
        vae_path = _get_model_path(vae_model, "vae")
        # Try text_encoders first, then clip
        text_encoder_path = _get_model_path(text_encoder, "text_encoders")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            text_encoder_path = _get_model_path(text_encoder, "clip")

        # Check if using folder path for images
        use_folder_path = False
        folder_images = []
        folder_captions = []

        if images_path and images_path.strip():
            images_path = os.path.expanduser(images_path.strip())
            if os.path.isdir(images_path):
                # Find all image files in the folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                for filename in sorted(os.listdir(images_path)):
                    if filename.lower().endswith(image_extensions):
                        img_path = os.path.join(images_path, filename)
                        folder_images.append(img_path)

                        # Look for matching caption file
                        base_name = os.path.splitext(filename)[0]
                        caption_file = os.path.join(images_path, f"{base_name}.txt")
                        if os.path.exists(caption_file):
                            with open(caption_file, 'r', encoding='utf-8') as f:
                                folder_captions.append(f.read().strip())
                        else:
                            folder_captions.append(caption)  # Use default caption

                if folder_images:
                    use_folder_path = True
                    print(f"[Musubi Z-Image] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[Musubi Z-Image] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[Musubi Z-Image] Invalid folder path: {images_path}, falling back to inputs")

        if not use_folder_path:
            # Collect all images and captions from inputs
            all_images = []
            all_captions = []

            if image_1 is not None:
                all_images.append(image_1)
                cap_1 = kwargs.get("caption_1", "")
                all_captions.append(cap_1 if cap_1 else caption)

            for i in range(2, inputcount + 1):
                img = kwargs.get(f"image_{i}")
                if img is not None:
                    all_images.append(img)
                    cap = kwargs.get(f"caption_{i}", "")
                    all_captions.append(cap if cap else caption)

            if not all_images:
                raise ValueError("No images provided. Either set images_path to a folder containing images, or connect at least one image input.")

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[Musubi Z-Image] Training with {num_images} image(s)")
        print(f"[Musubi Z-Image] DiT: {dit_model}")
        print(f"[Musubi Z-Image] VAE: {vae_model}")
        print(f"[Musubi Z-Image] Text Encoder: {text_encoder}")

        # Parse and validate device indexes for multi-GPU
        device_ids = []
        num_gpus = 0
        if enable_multi_gpu:
            if not device_indexes or not device_indexes.strip():
                raise ValueError("device_indexes must be provided when enable_multi_gpu is True. Example: '0,1' or '0,1,2'")
            try:
                device_ids, num_gpus = _parse_device_indexes(device_indexes)
                if num_gpus == 0:
                    raise ValueError("device_indexes must contain at least one GPU ID when enable_multi_gpu is True")
                print(f"[Musubi Z-Image] Multi-GPU enabled: Using GPUs {device_ids}")
            except ValueError as e:
                raise ValueError(f"Invalid device_indexes: {e}")

        # Get VRAM preset settings
        preset = MUSUBI_ZIMAGE_VRAM_PRESETS.get(vram_mode, MUSUBI_ZIMAGE_VRAM_PRESETS["Low (768px)"])
        print(f"[Musubi Z-Image] Using VRAM mode: {vram_mode}")

        # Validate paths
        accelerate_path = _get_accelerate_path(musubi_path)
        train_script = os.path.join(musubi_path, "src", "musubi_tuner", "zimage_train_network.py")
        convert_script = os.path.join(musubi_path, "src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")

        if not os.path.isdir(musubi_path):
            raise FileNotFoundError(
                f"musubi_path does not exist or is not a folder: {musubi_path}\n"
                "Set musubi_path to your musubi-tuner install root (the folder that contains \"src\\musubi_tuner\\...\")."
            )

        # accelerate is required for the default launcher, but NOT for our Windows multi-GPU path.
        if not (sys.platform == "win32" and enable_multi_gpu and num_gpus > 1):
            if not os.path.exists(accelerate_path):
                raise FileNotFoundError(f"Musubi Tuner accelerate not found at: {accelerate_path}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(
                f"zimage_train_network.py not found at: {train_script}\n"
                "This almost always means musubi_path is wrong.\n"
                "Example (Windows): musubi_path should look like \"E:\\AI\\musubi-tuner\" (your actual folder),\n"
                "so the script exists at <musubi_path>\\src\\musubi_tuner\\zimage_train_network.py."
            )
        if not os.path.exists(convert_script):
            raise FileNotFoundError(f"convert_z_image_lora_to_comfy.py not found at: {convert_script}")
        if not dit_path or not os.path.exists(dit_path):
            raise FileNotFoundError(f"DiT model not found at: {dit_path}")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found at: {vae_path}")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder not found at: {text_encoder_path}")

        # Save settings
        global _musubi_config
        _musubi_config['musubi_path'] = musubi_path
        _musubi_config['trainer_settings'] = {
            'dit_model': dit_model,
            'vae_model': vae_model,
            'text_encoder': text_encoder,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
            'enable_multi_gpu': enable_multi_gpu,
            'device_indexes': device_indexes,
            'reuse_existing_caches': reuse_existing_caches,
            'save_training_state': save_training_state,
            'continue_from_state': continue_from_state,
            'reset_existing_run': reset_existing_run,
            'checkpoint_every_n_steps': checkpoint_every_n_steps,
            'checkpoint_every_n_epochs': checkpoint_every_n_epochs,
            'warmup_steps': warmup_steps,
            'gradient_checkpointing': gradient_checkpointing,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'enable_bucket': enable_bucket,
            'bucket_no_upscale': bucket_no_upscale,
            'num_repeats': num_repeats,
        }
        _save_musubi_config()

        # Compute hash for caching
        if use_folder_path:
            image_hash = _compute_image_hash(folder_images, folder_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=True)
        else:
            image_hash = _compute_image_hash(all_images, all_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=False)

        # Check cache
        if keep_lora and image_hash in _musubi_lora_cache:
            cached_path = _musubi_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Musubi Z-Image] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _musubi_lora_cache[image_hash]
                _save_musubi_cache()

        # Per-LoRA output folder (stable by output_name)
        output_name = _validate_lora_name(output_name)
        output_root = os.path.join(musubi_path, "output")
        os.makedirs(output_root, exist_ok=True)

        lora_folder = os.path.join(output_root, output_name)
        lora_folder_exists = os.path.isdir(lora_folder)

        dataset_dir = os.path.join(lora_folder, "dataset")
        cache_dir = os.path.join(lora_folder, "cache")
        config_path = os.path.join(lora_folder, "dataset_config.toml")

        if continue_from_state and reset_existing_run:
            raise ValueError("reset_existing_run cannot be used together with continue_from_state. Choose one.")

        if lora_folder_exists:
            if not continue_from_state:
                if not reset_existing_run:
                    raise RuntimeError(
                        f"LoRA folder already exists for output_name='{output_name}': {lora_folder}\n"
                        "- To resume training, enable continue_from_state.\n"
                        "- To start fresh, enable reset_existing_run (this will delete dataset/cache/state folders)."
                    )
                _reset_lora_run_folder(lora_folder, dataset_dir, cache_dir)
        else:
            os.makedirs(lora_folder, exist_ok=True)

        run_name = output_name
        lora_output_path = os.path.join(lora_folder, f"{run_name}.safetensors")
        lora_comfy_path = os.path.join(lora_folder, f"{run_name}_comfy.safetensors")

        # Create / reuse dataset + config
        created_dataset = False
        needs_dataset_setup = (not os.path.exists(config_path)) or (not os.path.isdir(dataset_dir))
        
        # Check if existing config has stale absolute paths (e.g., after folder rename)
        # If so, we need to regenerate the config with current (relative) paths
        needs_config_refresh = False
        if os.path.exists(config_path) and os.path.isdir(dataset_dir):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_text = f.read()
                # Check if config uses absolute paths that don't exist
                # Old configs have absolute paths like "E:/old-path/output/..." 
                # New configs use relative paths like "./dataset"
                if 'image_directory = "./' not in config_text and 'image_directory = ".' not in config_text:
                    # Config uses absolute paths - check if they're valid
                    import re as _re
                    img_dir_match = _re.search(r'image_directory\s*=\s*"([^"]+)"', config_text)
                    if img_dir_match:
                        stored_img_dir = img_dir_match.group(1).replace('/', os.sep)
                        if not os.path.isdir(stored_img_dir):
                            print(f"[Musubi Z-Image] Config has stale absolute paths (folder was likely renamed)")
                            print(f"[Musubi Z-Image] Stored: {stored_img_dir}")
                            print(f"[Musubi Z-Image] Current: {dataset_dir}")
                            needs_config_refresh = True
            except Exception:
                pass
        
        if needs_config_refresh and not needs_dataset_setup:
            # Regenerate config with current relative paths (preserves existing cache)
            print(f"[Musubi Z-Image] Regenerating config with relative paths for portability...")
            config_content = generate_dataset_config(
                image_folder=dataset_dir,
                resolution=preset['resolution'],
                batch_size=batch_size,
                enable_bucket=enable_bucket,
                bucket_no_upscale=bucket_no_upscale,
                num_repeats=num_repeats,
                cache_directory=cache_dir,
                use_relative_paths=True,
                subprocess_cwd=musubi_path,
            )
            save_config(config_content, config_path)
            print(f"[Musubi Z-Image] Config refreshed: {config_path}")
        
        if needs_dataset_setup:
            created_dataset = True
            os.makedirs(dataset_dir, exist_ok=True)
            # If we (re)create the dataset, remove old caches (cache files are keyed by filename/size).
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
            # Ensure dataset_dir is empty-ish on fresh run
            for entry in os.listdir(dataset_dir):
                path = os.path.join(dataset_dir, entry)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(dataset_dir, f"image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    caption_path = os.path.join(dataset_dir, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    img_data = img_tensor[0]
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    image_path = os.path.join(dataset_dir, f"image_{idx+1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    caption_path = os.path.join(dataset_dir, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            print(f"[Musubi Z-Image] Saved {num_images} images to {dataset_dir}")

            # Generate dataset config (use relative paths for portability across folder renames)
            config_content = generate_dataset_config(
                image_folder=dataset_dir,
                resolution=preset['resolution'],
                batch_size=batch_size,
                enable_bucket=enable_bucket,
                bucket_no_upscale=bucket_no_upscale,
                num_repeats=num_repeats,
                cache_directory=cache_dir,
                use_relative_paths=True,
                subprocess_cwd=musubi_path,
            )

            save_config(config_content, config_path)
            print(f"[Musubi Z-Image] Dataset config saved to {config_path}")

        # Set up subprocess environment
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # Add musubi-tuner src to PYTHONPATH so the musubi_tuner module can be found
        musubi_src_dir = os.path.join(musubi_path, "src")
        existing_pythonpath = env.get('PYTHONPATH', '')
        if existing_pythonpath:
            env['PYTHONPATH'] = musubi_src_dir + os.pathsep + existing_pythonpath
        else:
            env['PYTHONPATH'] = musubi_src_dir

        # Set CUDA_VISIBLE_DEVICES for multi-GPU support
        if enable_multi_gpu and device_ids:
            cuda_devices = ','.join(str(d) for d in device_ids)
            env['CUDA_VISIBLE_DEVICES'] = cuda_devices
            print(f"[Musubi Z-Image] Setting CUDA_VISIBLE_DEVICES={cuda_devices}")

            if sys.platform == 'win32':
                env['USE_LIBUV'] = '0'
                env['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
                print(f"[Musubi Z-Image] Windows multi-GPU: USE_LIBUV=0, TORCH_DISTRIBUTED_BACKEND=gloo")

        # Use custom python exe if provided, otherwise detect from musubi_path
        if custom_python_exe and custom_python_exe.strip():
            python_path = custom_python_exe.strip()
            if not os.path.exists(python_path):
                raise FileNotFoundError(f"Custom python.exe not found at: {python_path}")
        else:
            python_path = _get_venv_python_path(musubi_path)

        # VAE preflight (fast failure): detect obvious wrong VAE checkpoints (e.g. WAN VAE) before caching
        # This runs in the musubi venv python so it can always import safetensors.
        if isinstance(vae_path, str) and vae_path.lower().endswith(".safetensors"):
            preflight_code = (
                "import sys\n"
                "from safetensors import safe_open\n"
                "p = sys.argv[1]\n"
                "with safe_open(p, framework='pt', device='cpu') as f:\n"
                "    keys = list(f.keys())\n"
                "def has(substr):\n"
                "    return any(substr in k for k in keys)\n"
                "ok = has('encoder.conv_in.weight') and has('decoder.conv_in.weight')\n"
                "if not ok:\n"
                "    print('VAE_PRECHECK_FAIL')\n"
                "    print('Missing expected key substrings: encoder.conv_in.weight / decoder.conv_in.weight')\n"
                "    print('first_keys=', keys[:25])\n"
                "    raise SystemExit(2)\n"
                "print('VAE_PRECHECK_OK')\n"
            )

            preflight_cmd = [python_path, "-c", preflight_code, vae_path]
            preflight_rc = _run_subprocess_with_cancel(
                preflight_cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner][preflight] ",
            )
            if preflight_rc != 0:
                raise RuntimeError(
                    "Selected VAE checkpoint does not look compatible with Z-Image AutoencoderKL.\n"
                    f"You selected: {vae_path}\n"
                    "This usually means you picked a WAN/other VAE. For Z-Image training, select the Z-Image/Flux VAE (e.g. flux\\ae.safetensors)."
                )

        # If resuming, pick newest state dir inside this LoRA folder
        resume_state_dir = ""
        resumed_step = 0  # Track resumed step for adjusting max_train_steps
        if continue_from_state and not created_dataset:
            resume_state_dir = _find_latest_state_dir(lora_folder)
            if resume_state_dir:
                print(f"[Musubi Z-Image] Resuming from state: {resume_state_dir}")
                # Parse step number from state directory name (e.g., "MyLora-step00001875-state")
                import re as _re
                _step_match = _re.search(r'-step(\d+)-state$', resume_state_dir)
                if _step_match:
                    resumed_step = int(_step_match.group(1))
                    print(f"[Musubi Z-Image] Detected resumed step: {resumed_step}")
            else:
                print(
                    f"[Musubi Z-Image] continue_from_state is enabled, but no '*-state' folders were found in: {lora_folder}. "
                    "Starting without resume."
                )
                if not save_training_state:
                    print("[Musubi Z-Image] Note: save_training_state is disabled, so no state will be saved for future resumes.")

        # Pre-cache latents and text encoder outputs (REQUIRED for Musubi Z-Image training)
        print(f"[Musubi Z-Image] Pre-caching latents and text encoder outputs...")

        # Cache latents / text encoder outputs
        cache_latents_script = os.path.join(musubi_path, "src", "musubi_tuner", "zimage_cache_latents.py")
        if not os.path.exists(cache_latents_script):
            raise FileNotFoundError(f"zimage_cache_latents.py not found at: {cache_latents_script}")

        cache_te_script = os.path.join(musubi_path, "src", "musubi_tuner", "zimage_cache_text_encoder_outputs.py")
        if not os.path.exists(cache_te_script):
            raise FileNotFoundError(f"zimage_cache_text_encoder_outputs.py not found at: {cache_te_script}")

        # Multi-GPU cache sharding:
        # Musubi's cache scripts remove cache files not in their dataset unless --keep_cache is passed.
        # When running multiple shards in parallel, each shard must use --keep_cache to avoid deleting
        # the other shards' cache files.
        use_sharded_cache = bool(enable_multi_gpu and num_gpus > 1 and device_ids)

        if use_sharded_cache:
            # Shard the CURRENT dataset folder, not the node inputs. This matches resume behavior.
            image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif")
            dataset_images = [
                f for f in sorted(os.listdir(dataset_dir)) if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith(image_exts)
            ]
            if not dataset_images:
                raise RuntimeError(f"No images found in dataset_dir for caching: {dataset_dir}")

            # Ensure basenames are unique so cache paths don't collide.
            basenames = [os.path.splitext(os.path.basename(f))[0] for f in dataset_images]
            if len(set(basenames)) != len(basenames):
                raise RuntimeError("Dataset contains duplicate image basenames; cache file names would collide.")

            shard_tmp = tempfile.mkdtemp(prefix="musubi_cache_shards_", dir=lora_folder)
            try:
                chunk = _ceil_div(len(dataset_images), num_gpus)

                shard_specs = []
                for shard_idx in range(num_gpus):
                    start = shard_idx * chunk
                    if start >= len(dataset_images):
                        break
                    end = min(start + chunk, len(dataset_images))
                    shard_files = dataset_images[start:end]
                    if not shard_files:
                        continue

                    shard_jsonl = os.path.join(shard_tmp, f"shard_{shard_idx:02d}.jsonl")
                    shard_toml = os.path.join(shard_tmp, f"shard_{shard_idx:02d}.toml")

                    # Build metadata JSONL pointing at the real dataset files (no copies).
                    with open(shard_jsonl, "w", encoding="utf-8") as jf:
                        for fname in shard_files:
                            img_path = os.path.join(dataset_dir, fname)
                            cap_path = os.path.join(dataset_dir, os.path.splitext(fname)[0] + ".txt")
                            if not os.path.exists(cap_path):
                                raise FileNotFoundError(f"Missing caption file for dataset image: {cap_path}")
                            with open(cap_path, "r", encoding="utf-8") as cf:
                                cap = cf.read().strip()
                            jf.write(json.dumps({"image_path": img_path, "caption": cap}, ensure_ascii=False) + "\n")

                    # Create a shard dataset config that shares the same cache directory.
                    # Use forward slashes for TOML paths on Windows.
                    shard_jsonl_esc = shard_jsonl.replace("\\", "/")
                    cache_dir_esc = cache_dir.replace("\\", "/")
                    shard_toml_text = (
                        "# Musubi Tuner Z-Image Dataset Config (cache shard)\n"
                        "# Generated by ComfyUI Musubi Z-Image LoRA Trainer\n\n"
                        "[general]\n"
                        f"resolution = [{preset['resolution']}, {preset['resolution']}]\n"
                        f"batch_size = {int(batch_size)}\n"
                        f"enable_bucket = {str(enable_bucket).lower()}\n"
                        f"bucket_no_upscale = {str(bucket_no_upscale).lower()}\n\n"
                        "[[datasets]]\n"
                        f'image_jsonl_file = "{shard_jsonl_esc}"\n'
                        f'cache_directory = "{cache_dir_esc}"\n'
                        f"num_repeats = {int(num_repeats)}\n"
                    )
                    with open(shard_toml, "w", encoding="utf-8") as tf:
                        tf.write(shard_toml_text)

                    gpu_id = int(device_ids[shard_idx])
                    shard_env = env.copy()
                    shard_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    shard_specs.append(
                        {
                            "shard_idx": shard_idx,
                            "gpu_id": gpu_id,
                            "env": shard_env,
                            "dataset_config": shard_toml,
                        }
                    )

                if not shard_specs:
                    raise RuntimeError("Multi-GPU cache sharding enabled, but no cache shards were created.")

                print(
                    f"[Musubi Z-Image] Multi-GPU cache sharding enabled: {len(shard_specs)} shard(s) across GPUs "
                    f"{[s['gpu_id'] for s in shard_specs]} (total images={len(dataset_images)})"
                )

                # Fast-path: Skip caching entirely if all cache files already exist
                num_latent_caches = len(glob.glob(os.path.join(cache_dir, "*_zi.safetensors")))
                num_te_caches = len(glob.glob(os.path.join(cache_dir, "*_zi_te.safetensors")))
                num_images = len(dataset_images)
                skip_caching = reuse_existing_caches and num_latent_caches >= num_images and num_te_caches >= num_images
                if skip_caching:
                    print(
                        f"[Musubi Z-Image] Skipping cache phase: {num_latent_caches} latent + {num_te_caches} TE cache files "
                        f"already exist for {num_images} images (reuse_existing_caches=True)"
                    )
                else:
                    # Cache latents in parallel (one shard per GPU).
                    print(f"[Musubi Z-Image] Caching VAE latents (sharded)...")
                    latents_jobs = []
                    for s in shard_specs:
                        cmd = [
                            python_path,
                            cache_latents_script,
                            f"--dataset_config={s['dataset_config']}",
                            f"--vae={vae_path}",
                            "--keep_cache",
                        ]
                        if reuse_existing_caches:
                            cmd.append("--skip_existing")
                        latents_jobs.append(
                            {
                                "cmd": cmd,
                                "env": s["env"],
                                "prefix": f"[musubi-tuner][cache_latents][gpu{ s['gpu_id'] }] ",
                            }
                        )

                    _run_parallel_subprocesses_with_cancel(latents_jobs, cwd=musubi_path, startupinfo=startupinfo)
                    print(f"[Musubi Z-Image] VAE latents cached (sharded).")

                    # Cache text encoder outputs in parallel (one shard per GPU).
                    print(f"[Musubi Z-Image] Caching text encoder outputs (sharded)...")
                    te_jobs = []
                    for s in shard_specs:
                        cmd = [
                            python_path,
                            cache_te_script,
                            f"--dataset_config={s['dataset_config']}",
                            f"--text_encoder={text_encoder_path}",
                            "--batch_size=1",
                            "--keep_cache",
                        ]
                        if reuse_existing_caches:
                            cmd.append("--skip_existing")
                        if preset.get("fp8_llm", False):
                            cmd.append("--fp8_llm")
                        te_jobs.append(
                            {
                                "cmd": cmd,
                                "env": s["env"],
                                "prefix": f"[musubi-tuner][cache_text][gpu{ s['gpu_id'] }] ",
                            }
                        )

                    _run_parallel_subprocesses_with_cancel(te_jobs, cwd=musubi_path, startupinfo=startupinfo)
                    print(f"[Musubi Z-Image] Text encoder outputs cached (sharded).")

            finally:
                try:
                    shutil.rmtree(shard_tmp)
                except Exception as e:
                    print(f"[Musubi Z-Image] Warning: could not remove cache shard temp dir: {shard_tmp}: {e}")

        else:
            # Single-GPU caching (original behavior)
            print(f"[Musubi Z-Image] Caching VAE latents...")
            cache_latents_cmd = [
                python_path,
                cache_latents_script,
                f"--dataset_config={config_path}",
                f"--vae={vae_path}",
            ]
            if reuse_existing_caches:
                cache_latents_cmd.append("--skip_existing")

            cache_latents_rc = _run_subprocess_with_cancel(
                cache_latents_cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner] ",
            )
            if cache_latents_rc != 0:
                raise RuntimeError(f"Latent caching failed with code {cache_latents_rc}")

            print(f"[Musubi Z-Image] VAE latents cached.")

            print(f"[Musubi Z-Image] Caching text encoder outputs...")
            cache_te_cmd = [
                python_path,
                cache_te_script,
                f"--dataset_config={config_path}",
                f"--text_encoder={text_encoder_path}",
                "--batch_size=1",
            ]
            if reuse_existing_caches:
                cache_te_cmd.append("--skip_existing")

            # Use fp8 for text encoder caching if enabled
            if preset.get('fp8_llm', False):
                cache_te_cmd.append("--fp8_llm")

            cache_te_rc = _run_subprocess_with_cancel(
                cache_te_cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner] ",
            )
            if cache_te_rc != 0:
                raise RuntimeError(f"Text encoder caching failed with code {cache_te_rc}")

            print(f"[Musubi Z-Image] Text encoder outputs cached.")

        # Build train args (shared)
        train_args = [
            f"--dit={dit_path}",
            f"--vae={vae_path}",
            f"--text_encoder={text_encoder_path}",
            f"--dataset_config={config_path}",
            "--sdpa",
            f"--mixed_precision={preset['mixed_precision']}",
            "--timestep_sampling=shift",
            "--weighting_scheme=none",
            "--discrete_flow_shift=2.0",
            f"--optimizer_type={preset['optimizer']}",
            f"--learning_rate={learning_rate}",
            f"--network_module=networks.lora_zimage",
            f"--network_dim={lora_rank}",
            f"--network_alpha={lora_rank}",
        ]
        
        # Adjust max_train_steps when resuming to account for already completed steps
        # (Musubi Tuner bug: step counter resets to 0 on resume, causing over-training)
        effective_training_steps = int(training_steps)
        if resumed_step > 0:
            remaining_steps = max(1, int(training_steps) - resumed_step)
            print(f"[Musubi Z-Image] Adjusting max_train_steps: {training_steps} - {resumed_step} (resumed) = {remaining_steps} remaining")
            effective_training_steps = remaining_steps
        train_args.append(f"--max_train_steps={effective_training_steps}")
        
        train_args.extend([
            "--max_data_loader_n_workers=2",
            "--persistent_data_loader_workers",
            f"--output_dir={lora_folder}",
            f"--output_name={run_name}",
            "--seed=42",
        ])

        if gradient_accumulation_steps and int(gradient_accumulation_steps) != 1:
            train_args.append(f"--gradient_accumulation_steps={int(gradient_accumulation_steps)}")

        # Windows multi-GPU: Musubi scales max_train_steps down by world size, so also scale step-based schedules
        # (warmup + step checkpointing) to keep the same cadence in terms of the user-entered training_steps.
        warmup_steps_eff = int(warmup_steps) if warmup_steps else 0
        checkpoint_steps_eff = int(checkpoint_every_n_steps) if checkpoint_every_n_steps else 0
        if sys.platform == "win32" and enable_multi_gpu and num_gpus > 1:
            world = int(num_gpus)

            if warmup_steps_eff > 0:
                orig = warmup_steps_eff
                warmup_steps_eff = max(1, _ceil_div(warmup_steps_eff, world))
                if warmup_steps_eff != orig:
                    print(
                        f"[Musubi Z-Image] Multi-GPU warmup scaling (Windows): lr_warmup_steps {orig} -> {warmup_steps_eff} (world_size={world})"
                    )

            if checkpoint_steps_eff > 0:
                input_steps = int(training_steps)
                desired_ckpts = input_steps // checkpoint_steps_eff
                # Only scale if the user would get at least one checkpoint on single-GPU.
                if desired_ckpts > 0:
                    orig = checkpoint_steps_eff
                    checkpoint_steps_eff = max(1, _ceil_div(checkpoint_steps_eff, world))
                    eff_steps = _ceil_div(input_steps, world)
                    if eff_steps // checkpoint_steps_eff < desired_ckpts:
                        checkpoint_steps_eff = max(1, eff_steps // desired_ckpts)
                    if checkpoint_steps_eff != orig:
                        print(
                            f"[Musubi Z-Image] Multi-GPU checkpoint scaling (Windows): save_every_n_steps {orig} -> {checkpoint_steps_eff} (world_size={world})"
                        )

        if warmup_steps_eff > 0:
            train_args.append("--lr_scheduler=constant_with_warmup")
            train_args.append(f"--lr_warmup_steps={warmup_steps_eff}")

        if checkpoint_steps_eff > 0:
            train_args.append(f"--save_every_n_steps={checkpoint_steps_eff}")
        if checkpoint_every_n_epochs and int(checkpoint_every_n_epochs) > 0:
            train_args.append(f"--save_every_n_epochs={int(checkpoint_every_n_epochs)}")

        if save_training_state:
            train_args.append("--save_state")

        if resume_state_dir:
            train_args.append(f"--resume={resume_state_dir}")

        # Memory optimization flags
        if gradient_checkpointing:
            train_args.append("--gradient_checkpointing")

        if preset['fp8_scaled']:
            train_args.append("--fp8_base")
            train_args.append("--fp8_scaled")

        if preset['fp8_llm']:
            train_args.append("--fp8_llm")

        if preset.get('blocks_to_swap', 0) > 0:
            train_args.append(f"--blocks_to_swap={preset['blocks_to_swap']}")

        # TensorBoard logging
        if enable_tensorboard:
            logging_dir = os.path.join(lora_folder, "logs")
            os.makedirs(logging_dir, exist_ok=True)
            train_args.append(f"--logging_dir={logging_dir}")
            train_args.append("--log_with=tensorboard")
            print(f"[Musubi Z-Image] TensorBoard logging enabled: {logging_dir}")
            print(f"[Musubi Z-Image] Run 'tensorboard --logdir \"{logging_dir}\" --port {tensorboard_port}' to view training progress")

        # Build training command
        cmd = [
            python_path,
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process=1",
            f"--mixed_precision={preset['mixed_precision']}",
            train_script,
        ]

        # Add multi-GPU flags only on non-Windows
        if enable_multi_gpu and num_gpus > 0 and sys.platform != 'win32':
            cmd.insert(4, f"--num_processes={num_gpus}")
            cmd.insert(5, "--multi_gpu")
            print(f"[Musubi Z-Image] Multi-GPU training: {num_gpus} processes on GPUs {device_ids}")

        cmd.extend(train_args)

        # Checkpoint/epoch math (approx, to reduce confusion)
        try:
            world = num_gpus if (enable_multi_gpu and num_gpus > 0) else 1
            # Use effective_training_steps (already adjusted for resume) as the base
            input_steps = int(effective_training_steps)
            effective_steps = input_steps
            # Musubi's Windows `--multi_gpu` trainer scales max_train_steps down by world size.
            if sys.platform == "win32" and enable_multi_gpu and world > 1:
                effective_steps = _ceil_div(input_steps, world)

            bs = max(1, int(batch_size))
            repeats = max(1, int(num_repeats))
            grad_accum = max(1, int(gradient_accumulation_steps)) if gradient_accumulation_steps else 1

            samples_per_epoch = max(1, int(num_images) * repeats)
            global_batches_per_epoch = _ceil_div(samples_per_epoch, bs)
            optim_steps_per_epoch = _ceil_div(global_batches_per_epoch, world * grad_accum)
            est_epochs = _ceil_div(max(1, effective_steps), max(1, optim_steps_per_epoch))

            if sys.platform == "win32" and enable_multi_gpu and world > 1 and effective_steps != input_steps:
                print(
                    f"[Musubi Z-Image] Multi-GPU step scaling (Windows): max_train_steps {input_steps} -> {effective_steps} (world_size={world})"
                )

            print(
                f"[Musubi Z-Image] Epoch math (approx): images={num_images}, num_repeats={repeats} => samples/epoch={samples_per_epoch}, "
                f"batch_size={bs}, world_size={world}, grad_accum={grad_accum} => optim_steps/epochâ‰ˆ{optim_steps_per_epoch}, "
                f"effective_stepsâ‰ˆ{effective_steps} => epochsâ‰ˆ{est_epochs}"
            )

            if checkpoint_every_n_epochs and int(checkpoint_every_n_epochs) > 0:
                every = int(checkpoint_every_n_epochs)
                # Musubi skips saving on the final epoch; estimate how many intermediate epoch checkpoints you'll get.
                est_epoch_ckpts = max(0, (est_epochs - 1) // max(1, every))
                if est_epoch_ckpts == 0:
                    print(
                        f"[Musubi Z-Image] Note: checkpoint_every_n_epochs={every} will produce 0 intermediate epoch checkpoints for ~{est_epochs} epoch(s) "
                        "(Musubi does not save on the final epoch; it always saves the final model at the end). "
                        "If you want multiple checkpoints, increase steps, reduce num_repeats, or use checkpoint_every_n_steps."
                    )
        except Exception as e:
            print(f"[Musubi Z-Image] Warning: could not compute epoch/checkpoint estimate: {e}")

        print(f"[Musubi Z-Image] Starting training: {run_name}")
        if resumed_step > 0:
            print(f"[Musubi Z-Image] Images: {num_images}, Steps: {effective_training_steps} (resumed from {resumed_step}, total target: {training_steps}), LR: {learning_rate}, Rank: {lora_rank}")
        else:
            print(f"[Musubi Z-Image] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")

        # Run training
        if sys.platform == 'win32' and enable_multi_gpu and num_gpus > 1:
            # Windows multi-GPU: use Musubi's internal `--multi_gpu` launcher (gloo).
            win_multi_cmd = [python_path, train_script, "--multi_gpu"] + train_args
            train_rc = _run_subprocess_with_cancel(
                win_multi_cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner] ",
            )
            if train_rc != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {train_rc}")
        else:
            train_rc = _run_subprocess_with_cancel(
                cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner] ",
            )
            if train_rc != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {train_rc}")

        print(f"[Musubi Z-Image] Training completed!")

        # Find the trained LoRA
        if not os.path.exists(lora_output_path):
            possible_files = [f for f in os.listdir(lora_folder) if f.startswith(run_name) and f.endswith('.safetensors') and '_comfy' not in f]
            possible_files = sorted(possible_files)
            if possible_files:
                lora_output_path = os.path.join(lora_folder, possible_files[-1])
            else:
                raise FileNotFoundError(f"No LoRA file found in {lora_folder}")

        print(f"[Musubi Z-Image] Found trained LoRA: {lora_output_path}")

        # Convert LoRAs (final + checkpoints) to ComfyUI format
        # Musubi saves intermediate checkpoints as raw (non-ComfyUI) LoRA safetensors (e.g. output_name-000001.safetensors).
        # Convert them alongside the final model so they can be loaded directly in ComfyUI.
        print(f"[Musubi Z-Image] Converting LoRAs (final + checkpoints) to ComfyUI format...")

        raw_lora_files = [
            f
            for f in os.listdir(lora_folder)
            if f.startswith(run_name) and f.endswith(".safetensors") and "_comfy" not in f
        ]
        raw_lora_files = sorted(raw_lora_files)
        if not raw_lora_files:
            raise FileNotFoundError(f"No LoRA .safetensors files found to convert in: {lora_folder}")

        converted = 0
        for fname in raw_lora_files:
            src_path = os.path.join(lora_folder, fname)
            dst_path = os.path.join(lora_folder, os.path.splitext(fname)[0] + "_comfy.safetensors")

            # Skip if destination is already up-to-date
            try:
                if os.path.exists(dst_path) and os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
                    continue
            except Exception:
                # If mtime comparison fails, just attempt conversion (idempotent)
                pass

            convert_cmd = [python_path, convert_script, src_path, dst_path]
            convert_rc = _run_subprocess_with_cancel(
                convert_cmd,
                cwd=musubi_path,
                env=env,
                startupinfo=startupinfo,
                prefix="[musubi-tuner] ",
            )
            if convert_rc != 0:
                raise RuntimeError(f"LoRA conversion failed with code {convert_rc} for: {src_path}")
            if not os.path.exists(dst_path):
                raise FileNotFoundError(f"Converted LoRA not found at: {dst_path}")
            converted += 1

        print(f"[Musubi Z-Image] Converted {converted} LoRA file(s) to ComfyUI format.")

        if not os.path.exists(lora_comfy_path):
            raise FileNotFoundError(f"Converted final LoRA not found at: {lora_comfy_path}")

        # Handle caching - cache the ComfyUI format LoRA
        if keep_lora:
            _musubi_lora_cache[image_hash] = lora_comfy_path
            _save_musubi_cache()
            print(f"[Musubi Z-Image] LoRA saved and cached at: {lora_comfy_path}")
        else:
            print(f"[Musubi Z-Image] LoRA available at: {lora_comfy_path}")

        return (lora_comfy_path,)

