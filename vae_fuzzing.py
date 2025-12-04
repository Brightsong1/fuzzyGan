import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim

from corpus_manager import (
    MAX_CORPUS_SIZE,
    MAX_INPUT_SIZE,
    clean_corpus_dir,
    load_corpus,
    save_and_log_corpus,
)
from fuzzer_runner import compute_coverage_loss, run_fuzzer
from policy import QLearningPolicy
from storage import open_storage
from syzkaller_adapter import workdir_from_config, load_manager_config
from vae_model import VAE, vae_loss

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def load_summary(summary_file: Path) -> Tuple[List[str], int, Dict[str, Dict]]:
    with summary_file.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    worth_fuzzing = [t["name"] for t in summary.get("targets", []) if t.get("worth_fuzzing")]
    function_map = {
        t["name"]: t
        for t in summary.get("targets", [])
        if t.get("worth_fuzzing") and t.get("seed_dir")
    }
    return worth_fuzzing, len(summary.get("targets", [])), function_map


def ensure_seed_corpus(func_info: Dict, corpus_dst: Path) -> Tuple[Path, List[Dict[str, bytes]]]:
    corpus_dst.mkdir(parents=True, exist_ok=True)
    corpus_backup = corpus_dst / "backup"
    seed_source = Path(func_info["seed_dir"])
    corpus_root = corpus_dst.parent
    fallback_dir = corpus_root / "seed_override"

    def _collect_files(directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file())

    seed_candidates: List[Path] = []
    for directory in (seed_source, fallback_dir):
        if not directory or not directory.exists():
            continue
        files = _collect_files(directory)
        if files:
            seed_candidates = files
            if directory == seed_source:
                break

    normalized_seeds: List[Tuple[str, bytes, bytes, int]] = []
    for index, seed_path in enumerate(seed_candidates[:MAX_CORPUS_SIZE]):
        raw = seed_path.read_bytes()
        if not raw:
            raw = b"\x00"
        trimmed = raw[:MAX_INPUT_SIZE]
        if len(trimmed) < MAX_INPUT_SIZE:
            trimmed = trimmed + b"\x00" * (MAX_INPUT_SIZE - len(trimmed))
        prefix = raw[:32] if raw else b"\x00"
        length = max(1, min(len(raw), MAX_INPUT_SIZE))
        name = seed_path.name if seed_path.suffix else f"seed{index:04d}.syzprog"
        normalized_seeds.append((name, trimmed, prefix, length))

    if not normalized_seeds:
        logging.warning("No initial seeds found for %s, creating default seed", func_info.get("name", "unknown"))
        raw = b"\x00"
        padded = raw + b"\x00" * (MAX_INPUT_SIZE - len(raw))
        normalized_seeds.append(("seed0000.syzprog", padded, raw, 1))

    seed_blueprint: List[Dict[str, bytes]] = [
        {"length": length, "prefix": prefix} for _, _, prefix, length in normalized_seeds
    ]

    backup_seeds = list(corpus_backup.glob("*")) if corpus_backup.exists() else []
    if not backup_seeds:
        corpus_backup.mkdir(parents=True, exist_ok=True)
        for name, data, _, _ in normalized_seeds:
            (corpus_backup / name).write_bytes(data)
        backup_seeds = list(corpus_backup.glob("*"))
        logging.info("Created corpus backup at %s", corpus_backup)

    if not any(corpus_dst.glob("*")):
        if backup_seeds:
            for seed in backup_seeds:
                shutil.copy2(seed, corpus_dst / seed.name)
        else:
            for name, data, _, _ in normalized_seeds:
                (corpus_dst / name).write_bytes(data)

    return corpus_backup, seed_blueprint


def train_for_function(
    args,
    storage,
    kernel: str,
    program_name: str,
    func_info: Dict,
    worth_fuzzing: List[str],
    total_functions: int,
    stats_path: Path,
    crash_dir: Path,
    corpus_dst: Path,
    epochs: int,
    fuzz_seconds: int,
    device: torch.device,
) -> None:
    corpus_backup, seed_blueprint = ensure_seed_corpus(func_info, corpus_dst)

    run_id = storage.start_run(
        "vae_fuzzing",
        kernel,
        program_name,
        {"corpus": str(corpus_dst), "epochs": epochs, "fuzz_seconds": fuzz_seconds},
    )
    try:
        vae = VAE().to(device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        policy = QLearningPolicy(storage, program_name)
        max_edges = 1

        start_time = time.time()
        for epoch in range(epochs):
            covered_funcs, covered_edges, observed_funcs = run_fuzzer(stats_path, crash_dir)
            clean_corpus_dir(corpus_dst)
            max_edges = max(max_edges, covered_edges or 1)
            coverage_loss_tensor, func_cov_ratio, code_cov_ratio = compute_coverage_loss(
                covered_funcs,
                observed_funcs,
                covered_edges,
                worth_fuzzing,
                total_functions,
                max_edges,
            )
            coverage_loss = coverage_loss_tensor.to(device)
            coverage_ratio = func_cov_ratio
            action = policy.select_action(coverage_ratio, covered_edges)
            mutation_scale = policy.mutation_scale(action)

            data = load_corpus(corpus_dst).to(device)
            if len(data) == 0:
                for seed in corpus_backup.glob("*"):
                    shutil.copy2(seed, corpus_dst / seed.name)
                data = load_corpus(corpus_dst).to(device)
                if len(data) == 0:
                    logging.warning("Empty corpus for %s, skipping epoch %d", program_name, epoch + 1)
                    continue

            optimizer.zero_grad()
            recon_data, mu, logvar = vae(data)
            loss = vae_loss(recon_data, data, mu, logvar, coverage_loss)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                z = torch.randn(len(data), vae.latent_dim).to(device) * mutation_scale
                new_data = vae.decoder(z).cpu()
                save_and_log_corpus(
                    new_data,
                    corpus_dst,
                    epoch + 1,
                    blueprint=seed_blueprint,
                    storage=storage,
                    run_id=run_id,
                    function=program_name,
                    extension=".syzprog",
                )

            reward = coverage_ratio + code_cov_ratio
            policy.update(coverage_ratio, covered_edges, reward)
            storage.record_epoch(
                run_id,
                program_name,
                epoch + 1,
                len(covered_funcs),
                covered_edges,
                float(loss.item()),
                metadata={"coverage_ratio": coverage_ratio, "code_coverage_ratio": code_cov_ratio},
            )

            elapsed = time.time() - start_time
            logging.info(
                "[%s] Epoch %d/%d | loss %.4f | coverage_loss %.4f | action=%s | elapsed %.1fs",
                program_name,
                epoch + 1,
                epochs,
                loss.item(),
                coverage_loss.item(),
                action,
                elapsed,
            )
        storage.finish_run(run_id, "completed")
    except Exception as exc:  # noqa: BLE001
        storage.finish_run(run_id, "failed", {"error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE-based adaptive fuzzing for syzkaller")
    parser.add_argument("--kernel", required=True, help="Kernel name (matches analysis summary)")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory containing analysis data")
    parser.add_argument("--manager-cfg", required=True, help="Path to syzkaller manager.cfg")
    parser.add_argument("--stats-file", help="Path to syzkaller fuzzer.stats (defaults to workdir/fuzzer.stats)")
    parser.add_argument("--crash-dir", help="Path to syzkaller crashes directory (defaults to workdir/crashes)")
    parser.add_argument(
        "--programs",
        help="Comma-separated list of program names to fuzz. Defaults to all worth-fuzzing entries in the summary.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Epochs per program (default: 300)")
    parser.add_argument("--fuzz-seconds", type=int, default=60, help="Seconds per epoch fuzzing window (default: 60)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of full cycles across selected programs")
    parser.add_argument("--fuzzer-prefix", default="fuzzer-0", help="Corpus directory under workdir/corpus/")
    parser.add_argument("--db-path", default="fuzz_out/fuzzygan.db", help="SQLite database path for run metadata")
    args = parser.parse_args()

    corpus_root = Path(args.out_dir) / args.kernel
    summary_path = corpus_root / "analysis_summary.json"
    if not summary_path.exists():
        logging.error("analysis_summary.json not found at %s", summary_path)
        return

    worth_fuzzing, total_functions, programs = load_summary(summary_path)
    if not programs:
        logging.error("No worth-fuzzing entries with seed data found in %s", summary_path)
        return

    if args.programs:
        requested = [name.strip() for name in args.programs.split(",") if name.strip()]
        target_functions = [name for name in requested if name in programs]
        missing = set(requested) - set(target_functions)
        for name in missing:
            logging.warning("Requested program %s not available; skipping", name)
    else:
        target_functions = list(programs.keys())

    if not target_functions:
        logging.error("No valid programs selected for fuzzing")
        return

    cfg = load_manager_config(Path(args.manager_cfg))
    workdir = workdir_from_config(cfg)
    stats_path = Path(args.stats_file).expanduser() if args.stats_file else workdir / "fuzzer.stats"
    crash_dir = Path(args.crash_dir).expanduser() if args.crash_dir else workdir / "crashes"
    corpus_base = workdir / "corpus" / args.fuzzer_prefix

    storage = open_storage(Path(args.db_path).expanduser())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for cycle in range(args.cycles):
            logging.info("=== Cycle %d/%d ===", cycle + 1, args.cycles)
            for program_name in target_functions:
                func_info = programs[program_name]
                try:
                    train_for_function(
                        args,
                        storage,
                        args.kernel,
                        program_name,
                        func_info,
                        worth_fuzzing,
                        total_functions,
                        stats_path,
                        crash_dir,
                        corpus_base,
                        args.epochs,
                        args.fuzz_seconds,
                        device,
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.exception("Error during fuzzing for %s: %s", program_name, exc)
                    continue
    finally:
        storage.close()


if __name__ == "__main__":
    main()
