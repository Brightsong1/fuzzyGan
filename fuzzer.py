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

def build_library(oss_fuzz_dir, library_name):
    project_dir = Path(oss_fuzz_dir) / "projects" / library_name
    build_script = project_dir / "build.sh"
    if not build_script.exists():
        logging.error(f"build.sh not found for {library_name} in {project_dir}")
        raise FileNotFoundError(f"build.sh not found for {library_name}")
    env = os.environ.copy()
    env["OUT"] = str(Path(oss_fuzz_dir) / "build" / "out" / library_name)
    run_cmd(["bash", build_script], cwd=project_dir, env=env)

def load_analysis_summary(out_dir: str, library_name: str) -> dict:
    summary_path = Path(out_dir) / library_name / "analysis_summary.json"
    if not summary_path.exists():
        logging.error(f"Analysis summary not found: {summary_path}")
        raise FileNotFoundError(f"Analysis summary not found for {library_name}")
    with summary_path.open('r') as f:
        return json.load(f)

def run_fuzzing(out_dir: str, library_name: str, oss_fuzz_dir: str):
    summary = load_analysis_summary(out_dir, library_name)
    oss_fuzz_project_dir = Path(oss_fuzz_dir) / "projects" / library_name
    for func in summary["functions"]:
        if not func.get("worth_fuzzing", False) or not func.get("harness_path"):
            continue
        harness_path = Path(func["harness_path"])
        fuzzer_binary = Path(out_dir) / library_name / f"fuzzer_{func['name']}"
        compile_cmd = ["clang", "-fsanitize=fuzzer,address", "-I", str(oss_fuzz_project_dir), "-L", str(Path(oss_fuzz_dir) / "build" / "out" / library_name), f"-l{library_name}", harness_path, "-o", fuzzer_binary]
        for attempt in range(3):
            try:
                run_cmd(compile_cmd)
                if oss_fuzz_project_dir.exists():
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

def list_libraries(oss_fuzz_dir: str) -> list:
    projects_dir = Path(oss_fuzz_dir) / "projects"
    if not projects_dir.exists():
        logging.error(f"OSS-Fuzz projects directory not found: {projects_dir}")
        return []
    return [d.name for d in projects_dir.iterdir() if d.is_dir() and (d / "build.sh").exists()]

def main():
    parser = argparse.ArgumentParser(description="Fuzzing tool")
    parser.add_argument("command", choices=["analyze", "fuzz", "list-libraries", "list-analyzed"], help="Command to execute")
    parser.add_argument("--library", help="Library name")
    parser.add_argument("--header-paths", help="Comma-separated header file paths")
    parser.add_argument("--exclude-prefixes", help="Comma-separated function prefixes to exclude")
    parser.add_argument("--skip-functions", help="Comma-separated functions to skip")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory")
    parser.add_argument("--oss-fuzz-dir", required=True, help="Path to OSS-Fuzz repository")
    args = parser.parse_args()

    if args.command == "list-libraries":
        libraries = list_libraries(args.oss_fuzz_dir)
        print("Available libraries:", ", ".join(libraries) if libraries else "None")
        return
    elif args.command == "list-analyzed":
        out_dir = Path(args.out_dir)
        analyzed = [d.name for d in out_dir.iterdir() if (d / "analysis_summary.json").exists()]
        print("Analyzed libraries:", ", ".join(analyzed) if analyzed else "None")
        return

    if not args.library:
        parser.error("Library name required for analyze and fuzz commands")
    if args.command == "analyze" and not args.header_paths:
        parser.error("Header paths required for analyze command")

    library_config = {
        "name": args.library,
        "header_paths": args.header_paths.split(","),
        "src_dir": str(Path(args.oss_fuzz_dir) / "projects" / args.library),
        "exclude_prefixes": args.exclude_prefixes.split(",") if args.exclude_prefixes else [],
        "skip_functions": args.skip_functions.split(",") if args.skip_functions else []
    }

    if args.command == "analyze":
        build_library(args.oss_fuzz_dir, args.library)
        from preanalyze import analyze_library
        analyze_library(library_config, args.out_dir)
    elif args.command == "fuzz":
        run_fuzzing(args.out_dir, args.library, args.oss_fuzz_dir)

if __name__ == "__main__":
    main()