import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from preanalyze import analyze_kernel
from storage import open_storage
from syzkaller_adapter import (
    corpus_root,
    load_manager_config,
    stage_corpus,
    summarize_feedback,
    workdir_from_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("fuzzygan.fuzzer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FuzzyGan syzkaller pipeline")
    parser.add_argument("command", choices=["analyze", "stage-corpus", "collect-stats", "one-shot"])
    parser.add_argument("--kernel-name", help="Kernel identifier (e.g., linux)")
    parser.add_argument("--kernel-src", help="Path to kernel source tree")
    parser.add_argument("--focus-dirs", help="Comma-separated subpaths to analyze (e.g., drivers/net,fs)")
    parser.add_argument("--max-files", type=int, default=12, help="Max files per target slice")
    parser.add_argument("--max-bytes", type=int, default=16000, help="Max bytes per target slice")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory for analysis artifacts")
    parser.add_argument("--db-path", default="fuzz_out/fuzzygan.db", help="SQLite database path")
    parser.add_argument("--manager-cfg", help="Path to syzkaller manager.cfg")
    parser.add_argument("--programs-dir", help="Override programs directory (defaults to fuzz_out/<kernel>/programs)")
    parser.add_argument("--fuzzer-prefix", default="fuzzer-0", help="Corpus destination under workdir/corpus/")
    parser.add_argument("--stats-file", help="Path to syzkaller fuzzer.stats (defaults to workdir/fuzzer.stats)")
    parser.add_argument("--crash-dir", help="Path to syzkaller crashes directory (defaults to workdir/crashes)")
    return parser.parse_args()


def run_analyze(args: argparse.Namespace) -> None:
    if not args.kernel_name or not args.kernel_src:
        raise SystemExit("--kernel-name and --kernel-src are required for analyze")
    focus_dirs = [p.strip() for p in (args.focus_dirs or "").split(",") if p.strip()]
    storage = open_storage(Path(args.db_path).expanduser())
    run_id = storage.start_run(
        "preanalyze",
        args.kernel_name,
        metadata={"out_dir": args.out_dir, "kernel_src": args.kernel_src, "focus_dirs": focus_dirs},
    )
    try:
        kernel_config: Dict = {
            "name": args.kernel_name,
            "kernel_src": args.kernel_src,
            "focus_dirs": focus_dirs,
            "max_files": args.max_files,
            "max_bytes": args.max_bytes,
        }
        analyze_kernel(kernel_config, args.out_dir, storage=storage, run_id=run_id)
        storage.finish_run(run_id, "completed")
    except Exception as exc:  
        storage.finish_run(run_id, "failed", {"error": str(exc)})
        storage.close()
        raise
    storage.close()


def run_stage_corpus(args: argparse.Namespace) -> None:
    if not args.kernel_name:
        raise SystemExit("--kernel-name is required for stage-corpus")
    if not args.manager_cfg:
        raise SystemExit("--manager-cfg is required for stage-corpus")
    cfg = load_manager_config(Path(args.manager_cfg))
    programs_dir = (
        Path(args.programs_dir).expanduser()
        if args.programs_dir
        else Path(args.out_dir) / args.kernel_name / "programs"
    )
    seeds_root = Path(args.out_dir) / args.kernel_name / "seeds"
    seed_dirs = list(seeds_root.iterdir()) if seeds_root.exists() else []
    sources = [programs_dir] + seed_dirs
    staged = stage_corpus(sources, cfg, fuzzer_prefix=args.fuzzer_prefix)
    LOG.info("Staged programs: %s", ", ".join(str(p) for p in staged))


def run_collect_stats(args: argparse.Namespace) -> None:
    if not args.manager_cfg:
        raise SystemExit("--manager-cfg is required for collect-stats")
    cfg = load_manager_config(Path(args.manager_cfg))
    workdir = workdir_from_config(cfg)
    stats_path = Path(args.stats_file).expanduser() if args.stats_file else workdir / "fuzzer.stats"
    crash_dir = Path(args.crash_dir).expanduser() if args.crash_dir else workdir / "crashes"
    covered_edges, covered_syscalls, raw = summarize_feedback(stats_path, crash_dir)
    summary = {
        "workdir": str(workdir),
        "stats_file": str(stats_path),
        "covered_edges": covered_edges,
        "covered_syscalls": covered_syscalls,
        "crash_count": raw.get("crash_count", 0),
        "crash_dirs": raw.get("crash_dirs", []),
        "raw": raw,
    }
    print(json.dumps(summary, indent=2))


def run_one_shot(args: argparse.Namespace) -> None:
    if not args.kernel_name:
        raise SystemExit("--kernel-name is required for one-shot")
    if not args.manager_cfg:
        raise SystemExit("--manager-cfg is required for one-shot")

    programs_dir = (
        Path(args.programs_dir).expanduser()
        if args.programs_dir
        else Path(args.out_dir) / args.kernel_name / "programs"
    )

    # If prebuilt programs exist, skip LLM preanalysis.
    has_prebuilt = programs_dir.exists() and any(programs_dir.iterdir())
    if not has_prebuilt:
        if not args.kernel_src:
            raise SystemExit("--kernel-src is required when no prebuilt programs are present")
        focus_dirs = [p.strip() for p in (args.focus_dirs or "").split(",") if p.strip()]
        storage = open_storage(Path(args.db_path).expanduser())
        run_id = storage.start_run(
            "preanalyze",
            args.kernel_name,
            metadata={"out_dir": args.out_dir, "kernel_src": args.kernel_src, "focus_dirs": focus_dirs},
        )
        try:
            kernel_config: Dict = {
                "name": args.kernel_name,
                "kernel_src": args.kernel_src,
                "focus_dirs": focus_dirs,
                "max_files": args.max_files,
                "max_bytes": args.max_bytes,
            }
            analyze_kernel(kernel_config, args.out_dir, storage=storage, run_id=run_id)
            storage.finish_run(run_id, "completed")
        except Exception as exc:  # noqa: BLE001
            storage.finish_run(run_id, "failed", {"error": str(exc)})
            storage.close()
            raise
        storage.close()

    cfg = load_manager_config(Path(args.manager_cfg))
    seeds_root = Path(args.out_dir) / args.kernel_name / "seeds"
    seed_dirs = list(seeds_root.iterdir()) if seeds_root.exists() else []
    sources = [programs_dir] + seed_dirs
    staged = stage_corpus(sources, cfg, fuzzer_prefix=args.fuzzer_prefix)
    LOG.info("One-shot staged %d files into %s/corpus/%s", len(staged), corpus_root(cfg), args.fuzzer_prefix)
    LOG.info("Start syzkaller with: ./bin/syz-manager -config %s", args.manager_cfg)


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "stage-corpus":
        run_stage_corpus(args)
    elif args.command == "collect-stats":
        run_collect_stats(args)
    elif args.command == "one-shot":
        run_one_shot(args)


if __name__ == "__main__":
    main()
