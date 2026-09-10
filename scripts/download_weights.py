#!/usr/bin/env python3
"""
Pneumonia Detection - Pretrained Weights Downloader
Downloads official release weights from GitHub Releases into models/current/
"""

import os
import sys
import hashlib
import urllib.request

RELEASE_TAG = "v1.0.0"
REPO = "Anurag-amrev-7557/Pneumonia-Detection-DL"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

MODELS = {
    "densenet121_best.h5": {
        "url": f"{BASE_URL}/densenet121_best.h5",
        "sha256": "9602bd80913f304391607f200d31c41542e947fc628035803e710eb7c8ccb318",
        "size_mb": 36.1,
    },
    "best_model.h5": {
        "url": f"{BASE_URL}/best_model.h5",
        "sha256": "eb9ac5cbef60356cad8ca0fb578b36df6f88ac96a859c358d8d93f26f586e330",
        "size_mb": 211.5,
    },
}

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:5.1f}% [{mb_down:6.1f}MB / {mb_total:6.1f}MB]")
        sys.stdout.flush()

def download_weights(target_dir: str = "models/current"):
    os.makedirs(target_dir, exist_ok=True)
    print(f"🏥 Clinical Model Weights Downloader ({RELEASE_TAG})")
    print(f"Target Directory: {os.path.abspath(target_dir)}\n")

    for filename, info in MODELS.items():
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path):
            print(f"🔍 Checking existing {filename}...")
            if compute_sha256(dest_path) == info["sha256"]:
                print(f"✅ {filename} already exists and passed SHA-256 verification. Skipping.\n")
                continue
            else:
                print(f"⚠️  {filename} checksum mismatch. Re-downloading...")

        print(f"📥 Downloading {filename} (~{info['size_mb']} MB) from GitHub Releases...")
        try:
            urllib.request.urlretrieve(info["url"], dest_path, reporthook=reporthook)
            sys.stdout.write("\n")
            print(f"🔒 Verifying SHA-256 checksum...")
            file_sha = compute_sha256(dest_path)
            if file_sha == info["sha256"]:
                print(f"✅ Verified: {filename}\n")
            else:
                print(f"❌ Warning: Checksum mismatch for {filename}!")
        except Exception as e:
            print(f"\n❌ Error downloading {filename}: {e}\n")

    print("🎉 Pretrained weights download and verification complete.")

if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "models/current"
    download_weights(dest)
