import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.optim as optim

from corpus_manager import MAX_INPUT_SIZE, clean_corpus_dir, load_corpus, save_and_log_corpus
from fuzzer_runner import compute_coverage_loss, run_fuzzer
from policy import QLearningPolicy
from storage import open_storage
from vae_model import VAE, vae_loss

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def load_summary(summary_file: Path) -> Tuple[List[str], int, Dict[str, Dict]]:
    with summary_file.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    worth_fuzzing = [f["name"] for f in summary["functions"] if f.get("worth_fuzzing")]
    function_map = {
        f["name"]: f
        for f in summary["functions"]
        if f.get("worth_fuzzing") and f.get("seed_dir")
    }
    return worth_fuzzing, len(summary["functions"]), function_map


def ensure_seed_corpus(func_info: Dict, corpus_dst: Path) -> Tuple[Path, List[Dict[str, bytes]]]:
    corpus_dst.mkdir(parents=True, exist_ok=True)
    corpus_backup = corpus_dst / "backup"
    seed_source = Path(func_info["seed_dir"])
    seed_files = sorted(seed_source.glob("*.bin"))
    seed_blueprint: List[Dict[str, bytes]] = []

    backup_seeds: List[Path] = []
    if corpus_backup.exists():
        backup_seeds = list(corpus_backup.glob("*.bin"))

    if seed_files and not backup_seeds:
        corpus_backup.mkdir(parents=True, exist_ok=True)
        for seed in seed_files:
            target = corpus_backup / seed.name
            data = seed.read_bytes()
            if len(data) < MAX_INPUT_SIZE:
                data = data + b"\x00" * (MAX_INPUT_SIZE - len(data))
            target.write_bytes(data)
        backup_seeds = list(corpus_backup.glob("*.bin"))
        logging.info("Created corpus backup at %s", corpus_backup)

    if seed_files:
        for seed in seed_files:
            data = seed.read_bytes()
            prefix = data[:8] if len(data) >= 8 else data
            seed_blueprint.append({"length": MAX_INPUT_SIZE, "prefix": prefix})

    if not any(corpus_dst.glob("*.bin")):
        source_dirs: Iterable[Path] = []
        if backup_seeds:
            source_dirs = backup_seeds
        elif seed_files:
            source_dirs = list(seed_files)
        for seed in source_dirs:
            target = corpus_dst / Path(seed).name
            data = Path(seed).read_bytes()
            if len(data) < MAX_INPUT_SIZE:
                data = data + b"\x00" * (MAX_INPUT_SIZE - len(data))
            target.write_bytes(data)

    for seed in corpus_dst.glob("*.bin"):
        data = seed.read_bytes()
        if len(data) < MAX_INPUT_SIZE:
            seed.write_bytes(data + b"\x00" * (MAX_INPUT_SIZE - len(data)))

    return corpus_backup, seed_blueprint


def train_for_function(
    args,
    storage,
    library: str,
    function_name: str,
    func_info: Dict,
    worth_fuzzing: List[str],
    total_functions: int,
    out_dir: Path,
    epochs: int,
    fuzz_seconds: int,
    device: torch.device,
) -> None:
    candidate_names = [function_name, f"{function_name}_fuzzer"]
    fuzzer_path = None
    for name in candidate_names:
        path = out_dir / name
        if path.exists():
            fuzzer_path = path.resolve()
            break
    if fuzzer_path is None:
        logging.error("No fuzzer binary found for %s (tried: %s)", function_name, ", ".join(candidate_names))
        return
    logging.info("Using fuzzer binary %s for %s", fuzzer_path, function_name)

    corpus_root = Path(args.out_dir) / args.library
    corpus_dst = corpus_root / f"corpus_{function_name}"
    corpus_backup, seed_blueprint = ensure_seed_corpus(func_info, corpus_dst)

    run_id = storage.start_run(
        "vae_fuzzing",
        library,
        function_name,
        {"fuzzer_path": str(fuzzer_path), "epochs": epochs, "fuzz_seconds": fuzz_seconds},
    )
    try:
        vae = VAE().to(device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        policy = QLearningPolicy(storage, function_name)
        max_edges = 1

        start_time = time.time()
        for epoch in range(epochs):
            covered_funcs, covered_edges, observed_funcs = run_fuzzer(fuzzer_path, corpus_dst, fuzz_seconds)
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
                for seed in corpus_backup.glob("*.bin"):
                    shutil.copy2(seed, corpus_dst / seed.name)
                data = load_corpus(corpus_dst).to(device)
                if len(data) == 0:
                    logging.warning("Empty corpus for %s, skipping epoch %d", function_name, epoch + 1)
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
                    function=function_name,
                )

            reward = coverage_ratio + code_cov_ratio
            policy.update(coverage_ratio, covered_edges, reward)
            storage.record_epoch(
                run_id,
                function_name,
                epoch + 1,
                len(covered_funcs),
                covered_edges,
                float(loss.item()),
            )

            elapsed = time.time() - start_time
            logging.info(
                "[%s] Epoch %d/%d | loss %.4f | coverage_loss %.4f | action=%s | elapsed %.1fs",
                function_name,
                epoch + 1,
                epochs,
                loss.item(),
                coverage_loss.item(),
                action,
                elapsed,
            )
        storage.finish_run(run_id, "completed")
    except Exception as exc:  
        storage.finish_run(run_id, "failed", {"error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE-based adaptive fuzzing")
    parser.add_argument("--library", required=True, help="Library name")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory containing analysis data")
    parser.add_argument("--oss-fuzz-dir", help="Path to OSS-Fuzz repository (default: ~/Desktop/trspo/nsw/oss-fuzz)")
    parser.add_argument("--db-path", default="fuzz_out/fuzzygan.db", help="SQLite database path for run metadata")
    parser.add_argument(
        "--functions",
        help="Comma-separated list of functions to fuzz. Defaults to all worth-fuzzing entries in the summary.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Epochs per function (default: 300)")
    parser.add_argument("--fuzz-seconds", type=int, default=60, help="Seconds per epoch fuzzing run (default: 60)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of full cycles across selected functions")
    args = parser.parse_args()

    corpus_root = Path(args.out_dir) / args.library
    summary_path = corpus_root / "analysis_summary.json"
    if not summary_path.exists():
        logging.error("analysis_summary.json not found at %s", summary_path)
        return

    worth_fuzzing, total_functions, functions = load_summary(summary_path)
    if not functions:
        logging.error("No worth-fuzzing functions with seed data found in %s", summary_path)
        return

    if args.functions:
        requested = [name.strip() for name in args.functions.split(",") if name.strip()]
        target_functions = [name for name in requested if name in functions]
        missing = set(requested) - set(target_functions)
        for name in missing:
            logging.warning("Requested function %s not available; skipping", name)
    else:
        target_functions = list(functions.keys())

    if not target_functions:
        logging.error("No valid functions selected for fuzzing")
        return

    oss_fuzz_dir = Path(args.oss_fuzz_dir or "~/Desktop/trspo/nsw/oss-fuzz").expanduser()
    out_dir = oss_fuzz_dir / "build" / "out" / args.library
    if not out_dir.exists():
        logging.error("OSS-Fuzz build output not found at %s", out_dir)
        return

    storage = open_storage(Path(args.db_path).expanduser())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for cycle in range(args.cycles):
            logging.info("=== Cycle %d/%d ===", cycle + 1, args.cycles)
            for function_name in target_functions:
                func_info = functions[function_name]
                try:
                    train_for_function(
                        args,
                        storage,
                        args.library,
                        function_name,
                        func_info,
                        worth_fuzzing,
                        total_functions,
                        out_dir,
                        args.epochs,
                        args.fuzz_seconds,
                        device,
                    )
                except Exception as exc:  
                    logging.exception("Error during fuzzing for %s: %s", function_name, exc)
                    continue
    finally:
        storage.close()


if __name__ == "__main__":
    main()
