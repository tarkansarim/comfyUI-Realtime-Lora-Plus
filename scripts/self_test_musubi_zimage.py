"""Lightweight self-test for the ComfyUI Musubi Z-Image trainer node.

This does NOT run training.
It verifies:
- the node files parse (py_compile)
- dataset_config generation includes the expected fields
- (optional) the given musubi_path has the required musubi-tuner scripts

Usage:
  python scripts/self_test_musubi_zimage.py --musubi-path E:\\AI\\musubi-tuner
"""

from __future__ import annotations

import argparse
import os
import sys
import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]



def find_musubi_python(musubi_root: Path) -> Path | None:
        candidates = []
        if os.name == 'nt':
            candidates += [musubi_root / '.venv' / 'Scripts' / 'python.exe', musubi_root / 'venv' / 'Scripts' / 'python.exe']
        else:
            candidates += [musubi_root / '.venv' / 'bin' / 'python', musubi_root / 'venv' / 'bin' / 'python']
        for c in candidates:
            if c.exists():
                return c
        return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musubi-path", type=str, default=None, help="Path to musubi-tuner root (optional)")
    ap.add_argument("--vae-path", type=str, default=None, help="Optional: path to a VAE .safetensors to sanity-check for Z-Image compatibility")
    args = ap.parse_args()

    trainer_py = REPO_ROOT / "musubi_zimage_lora_trainer.py"
    cfg_py = REPO_ROOT / "musubi_zimage_config_template.py"

    print(f"[self-test] repo: {REPO_ROOT}")
    print("[self-test] compile-check...")
    py_compile.compile(str(trainer_py), doraise=True)
    py_compile.compile(str(cfg_py), doraise=True)

    print("[self-test] config template output...")
    # Import only the config template (safe, no ComfyUI deps)
    import importlib.util

    spec = importlib.util.spec_from_file_location("musubi_zimage_config_template", cfg_py)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    toml_text = mod.generate_dataset_config(
        image_folder="C:\\tmp\\dataset",
        resolution=768,
        batch_size=2,
        enable_bucket=True,
        bucket_no_upscale=True,
        num_repeats=10,
        cache_directory="C:\\tmp\\cache",
    )

    required_lines = [
        "[general]",
        "enable_bucket = true",
        "bucket_no_upscale = true",
        "batch_size = 2",
        "[[datasets]]",
        "cache_directory = \"C:/tmp/cache\"",
    ]

    missing = [ln for ln in required_lines if ln not in toml_text]
    if missing:
        print("[self-test] FAIL: dataset_config missing lines:")
        for ln in missing:
            print("  -", ln)
        return 2

    print("[self-test] OK: dataset_config contains expected fields")

    if args.musubi_path:
        mp = Path(args.musubi_path)
        print(f"[self-test] musubi-path check: {mp}")
        if not mp.is_dir():
            print("[self-test] FAIL: musubi-path is not a directory")
            return 3

        needed = [
            mp / "src" / "musubi_tuner" / "zimage_train_network.py",
            mp / "src" / "musubi_tuner" / "zimage_cache_latents.py",
            mp / "src" / "musubi_tuner" / "zimage_cache_text_encoder_outputs.py",
            mp / "src" / "musubi_tuner" / "networks" / "convert_z_image_lora_to_comfy.py",
        ]
        missing_files = [p for p in needed if not p.exists()]
        if missing_files:
            print("[self-test] FAIL: musubi-tuner files missing:")
            for p in missing_files:
                print("  -", p)
            return 4

        print("[self-test] OK: musubi-tuner scripts exist")


        if args.vae_path:
            vae_path = Path(args.vae_path)
            print(f"[self-test] vae-path check: {vae_path}")
            if not vae_path.exists():
                print("[self-test] FAIL: vae-path does not exist")
                return 5

            py = find_musubi_python(mp)
            if py is None:
                print("[self-test] FAIL: could not locate musubi venv python under musubi-path (.venv/venv)")
                return 6

            code = (
                "import sys\n"
                "from safetensors import safe_open\n"
                "p = sys.argv[1]\n"
                "with safe_open(p, framework='pt', device='cpu') as f:\n"
                "    keys = list(f.keys())\n"
                "def has(substr):\n"
                "    return any(substr in k for k in keys)\n"
                "ok = has('encoder.conv_in.weight') and has('decoder.conv_in.weight')\n"
                "print('OK' if ok else 'FAIL')\n"
                "if not ok:\n"
                "    print('first_keys=', keys[:25])\n"
                "    raise SystemExit(2)\n"
            )

            import subprocess
            res = subprocess.run([str(py), '-c', code, str(vae_path)], capture_output=True, text=True)
            out = (res.stdout or '').strip()
            err = (res.stderr or '').strip()
            if res.returncode != 0:
                print("[self-test] FAIL: VAE does not look Z-Image compatible")
                if out:
                    print(out)
                if err:
                    print(err)
                return 7
            print("[self-test] OK: VAE looks Z-Image compatible")

    print("[self-test] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except py_compile.PyCompileError as e:
        print("[self-test] FAIL: syntax error:")
        print(e)
        raise
