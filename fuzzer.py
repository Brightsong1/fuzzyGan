import os
import subprocess
import argparse
import json
import shutil
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_cmd(cmd, cwd=None, env=None, timeout=300):
    try:
        result = subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True, cwd=cwd, env=env, timeout=timeout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command {' '.join(cmd)} failed: {e.stderr}")
        raise

def build_library(src_dir, build_commands, env_vars):
    src_dir = Path(src_dir)
    env = os.environ.copy()
    env.update(env_vars)
    for cmd in build_commands:
        run_cmd(cmd, cwd=src_dir, env=env)

def load_analysis_summary(out_dir: str, library_name: str) -> dict:
    summary_path = Path(out_dir) / library_name / "analysis_summary.json"
    with summary_path.open('r') as f:
        return json.load(f)

def run_fuzzing(out_dir: str, library_name: str, src_dir: str, oss_fuzz_dir: str = None):
    summary = load_analysis_summary(out_dir, library_name)
    oss_fuzz_project_dir = Path(oss_fuzz_dir) / "projects" / library_name if oss_fuzz_dir else None
    for func in summary["functions"]:
        if not func.get("worth_fuzzing", False) or not func.get("harness_path"):
            continue
        harness_path = Path(func["harness_path"])
        fuzzer_binary = Path(out_dir) / library_name / f"fuzzer_{func['name']}"
        compile_cmd = ["clang", "-fsanitize=fuzzer,address", "-I", src_dir, "-L", src_dir, f"-l{library_name}", harness_path, "-o", fuzzer_binary]
        for attempt in range(3):
            try:
                run_cmd(compile_cmd)
                if oss_fuzz_project_dir:
                    shutil.copy(harness_path, oss_fuzz_project_dir / f"{func['name']}_fuzzer.c")
                    for seed in glob.glob(f"{func['seed_dir']}/*.bin"):
                        shutil.copy(seed, oss_fuzz_project_dir / Path(seed).name)
                subprocess.run([fuzzer_binary, func["seed_dir"]], timeout=300)
                break
            except subprocess.CalledProcessError as e:
                logging.error(f"Compilation failed for {func['name']}, attempt {attempt + 1}: {e}")
                if attempt < 2:
                    logging.info("Retrying compilation...")
                    continue
                logging.error(f"Failed to compile {func['name']} after 3 attempts")
                break

def main():
    parser = argparse.ArgumentParser(description="Fuzzing tool")
    parser.add_argument("command", choices=["analyze", "fuzz", "list-libraries", "list-analyzed"], help="Command to execute")
    parser.add_argument("--library", help="Library name")
    parser.add_argument("--header-paths", help="Comma-separated header file paths")
    parser.add_argument("--src-dir", help="Source directory")
    parser.add_argument("--build-commands", help="JSON string of build commands")
    parser.add_argument("--env-vars", help="JSON string of environment variables")
    parser.add_argument("--exclude-prefixes", help="Comma-separated function prefixes to exclude")
    parser.add_argument("--skip-functions", help="Comma-separated functions to skip")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory")
    parser.add_argument("--oss-fuzz-dir", help="Path to OSS-Fuzz repository")
    args = parser.parse_args()

    if args.command == "list-libraries":
        print("No predefined libraries; specify via --library and parameters")
        return
    elif args.command == "list-analyzed":
        out_dir = Path(args.out_dir)
        analyzed = [d.name for d in out_dir.iterdir() if (d / "analysis_summary.json").exists()]
        print("Analyzed libraries:", ", ".join(analyzed) if analyzed else "None")
        return

    if not all([args.library, args.header_paths, args.src_dir, args.build_commands]):
        parser.error("Library name, header paths, source dir, and build commands required for analyze/fuzz")

    library_config = {
        "name": args.library,
        "header_paths": args.header_paths.split(","),
        "src_dir": args.src_dir,
        "build_commands": json.loads(args.build_commands),
        "env": json.loads(args.env_vars) if args.env_vars else {},
        "exclude_prefixes": args.exclude_prefixes.split(",") if args.exclude_prefixes else [],
        "skip_functions": args.skip_functions.split(",") if args.skip_functions else []
    }

    if args.command == "analyze":
        from preanalyze import analyze_library
        analyze_library(library_config, args.out_dir)
    elif args.command == "fuzz":
        run_fuzzing(args.out_dir, args.library, args.src_dir, args.oss_fuzz_dir)

if __name__ == "__main__":
    main()