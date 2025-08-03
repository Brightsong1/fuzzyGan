import glob
import os
import shutil
import numpy as np
from pathlib import Path
import binascii
import zipfile
import logging

MAX_INPUT_SIZE = 1024
MAX_SEEDS_TO_LOG = 5
MAX_BYTES_TO_LOG = 64
MAX_CORPUS_SIZE = 271

def load_corpus(corpus_dir):
    data = []
    for f in glob.glob(str(corpus_dir / "*.bin")):
        with open(f, 'rb') as bf:
            bytes_data = bf.read(MAX_INPUT_SIZE)
            if len(bytes_data) < MAX_INPUT_SIZE:
                bytes_data += b'\x00' * (MAX_INPUT_SIZE - len(bytes_data))
            else:
                bytes_data = bytes_data[:MAX_INPUT_SIZE]
            data.append(np.frombuffer(bytes_data, dtype=np.uint8) / 255.0)
    return torch.from_numpy(np.array(data, dtype=np.float32)) if data else torch.tensor([])

def clean_corpus_dir(corpus_dir):
    for item in corpus_dir.iterdir():
        if item.is_file() and item.suffix != '.bin':
            item.unlink()

def save_and_log_corpus(data, corpus_dir, epoch, prefix="seed"):
    epoch_dir = corpus_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(exist_ok=True)
    existing_seeds = sorted(glob.glob(str(corpus_dir / "*.bin")))
    num_to_save = min(len(data), MAX_CORPUS_SIZE)
    seed_indices = list(range(min(num_to_save, len(existing_seeds)))) + list(range(len(existing_seeds), min(num_to_save, MAX_CORPUS_SIZE)))
    for i, idx in enumerate(seed_indices[:num_to_save]):
        bytes_data = (data[i] * 255).byte().numpy()
        seed_path = corpus_dir / f"{prefix}{idx}.bin" if idx >= len(existing_seeds) else Path(existing_seeds[idx])
        epoch_seed_path = epoch_dir / f"{prefix}{idx}_epoch{epoch}.bin"
        with open(epoch_seed_path, 'wb') as f:
            f.write(bytes_data)
        shutil.copy(epoch_seed_path, seed_path)
    for seed_file in glob.glob(str(epoch_dir / f"{prefix}*_epoch{epoch}.bin"))[:MAX_SEEDS_TO_LOG]:
        with open(seed_file, 'rb') as f:
            hex_data = binascii.hexlify(f.read(MAX_BYTES_TO_LOG)).decode('ascii') + ("..." if os.path.getsize(seed_file) > MAX_BYTES_TO_LOG else "")
            logging.info(f"Mutated seed {os.path.basename(seed_file)}: 0x{hex_data}")
    zip_path = corpus_dir.parent / "seed_corpus.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(str(corpus_dir / "*.bin")):
            zf.write(f, os.path.basename(f))