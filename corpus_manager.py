import logging
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch

from io import BytesIO

MAX_INPUT_SIZE = 1024
MAX_SEEDS_TO_LOG = 5
MAX_BYTES_TO_LOG = 64
MAX_CORPUS_SIZE = 271

LOG = logging.getLogger("fuzzygan.corpus")


def load_corpus(corpus_dir: Path) -> torch.Tensor:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tensors: List[np.ndarray] = []
    for seed_file in sorted(corpus_dir.glob("*.bin")):
        raw = seed_file.read_bytes()[:MAX_INPUT_SIZE]
        if len(raw) < MAX_INPUT_SIZE:
            raw += b"\x00" * (MAX_INPUT_SIZE - len(raw))
        tensors.append(np.frombuffer(raw, dtype=np.uint8) / 255.0)
    if not tensors:
        return torch.empty(0)
    return torch.from_numpy(np.stack(tensors, axis=0).astype(np.float32))


def clean_corpus_dir(corpus_dir: Path) -> None:
    for item in corpus_dir.iterdir():
        if item.is_file() and item.suffix != ".bin":
            item.unlink()


def _log_seed_samples(samples: List[tuple]) -> None:
    for name, content in samples[:MAX_SEEDS_TO_LOG]:
        data = content[:MAX_BYTES_TO_LOG]
        tail = "..." if len(content) > MAX_BYTES_TO_LOG else ""
        LOG.info("Mutated seed %s: 0x%s%s", name, data.hex(), tail)
def _apply_blueprint(raw: bytearray, blueprint: dict | None) -> bytes:
    if not blueprint:
        return bytes(raw)
    prefix: bytes = blueprint.get("prefix", b"") or b""
    length: int = blueprint.get("length", len(raw))
    if length < len(raw):
        raw = raw[:length]
    elif length > len(raw):
        raw.extend(b"\x00" * (length - len(raw)))
    if raw:
        enforce_len = len(prefix)
        if enforce_len >= len(raw) and len(raw) > 1:
            enforce_len = len(raw) - 1
        enforce_len = min(enforce_len, len(raw))
        if enforce_len > 0:
            raw[:enforce_len] = prefix[:enforce_len]
    if length < len(raw):
        return bytes(raw[:length])
    return bytes(raw)


def save_and_log_corpus(
    data: torch.Tensor,
    corpus_dir: Path,
    epoch: int,
    prefix: str = "seed",
    blueprint: List[dict] | None = None,
    storage=None,
    run_id: int | None = None,
    function: str | None = None,
) -> None:
    if data.numel() == 0:
        return
    corpus_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(data.shape[0], MAX_CORPUS_SIZE)
    sample_bytes: List[tuple] = []
    for index in range(num_samples):
        seed_name = f"{prefix}{index:04d}.bin"
        tensor = data[index]
        target_path = corpus_dir / seed_name
        raw = bytearray((tensor.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy().tobytes())
        applied = _apply_blueprint(raw, blueprint[index % len(blueprint)] if blueprint else None)
        target_path.write_bytes(applied)
        sample_bytes.append((seed_name, applied))

    for stale in sorted(corpus_dir.glob(f"{prefix}*.bin"))[num_samples:]:
        stale.unlink()

    _log_seed_samples(sample_bytes)

    if storage and run_id is not None and function:
        try:
            archive = BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, content in sample_bytes:
                    zf.writestr(name, content)
            storage.record_corpus_blob(run_id, function, epoch, archive.getvalue())
        except AttributeError:
            pass
