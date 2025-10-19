import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from oss_fuzz_interceptor import intercept
from storage import open_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("fuzzygan.fuzzer")


@dataclass
class LibraryConfig:
    name: str
    header_paths: List[Path]
    src_dir: Path
    exclude_prefixes: List[str]
    skip_functions: List[str]

    @classmethod
    def from_args(cls, args: argparse.Namespace, oss_fuzz_dir: Path) -> "LibraryConfig":
        header_paths = [Path(p).expanduser() for p in (args.header_paths or "").split(",") if p]
        return cls(
            name=args.library,
            header_paths=header_paths,
            src_dir=oss_fuzz_dir / "projects" / args.library,
            exclude_prefixes=[p.strip() for p in (args.exclude_prefixes or "").split(",") if p.strip()],
            skip_functions=[p.strip() for p in (args.skip_functions or "").split(",") if p.strip()],
        )


def run_cmd(cmd: Iterable[object], *, cwd: Optional[Path] = None, env: Optional[dict] = None, timeout: int = 300) -> str:
    cmd_list = [str(c) for c in cmd]
    display = " ".join(cmd_list)
    try:
        result = subprocess.run(cmd_list, check=True, text=True, capture_output=True, cwd=cwd, env=env, timeout=timeout)
        if result.stdout:
            LOG.debug(result.stdout)
        if result.stderr:
            LOG.debug(result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as error:
        LOG.error("Command failed: %s", display)
        if error.stdout:
            LOG.error("stdout:\n%s", error.stdout)
        if error.stderr:
            LOG.error("stderr:\n%s", error.stderr)
        raise


def get_build_out_dir(oss_fuzz_dir: Path, library_name: str) -> Path:
    return oss_fuzz_dir / "build" / "out" / library_name


def ensure_build_dirs(out_dir: Path, library_out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    library_out_dir.mkdir(parents=True, exist_ok=True)


def setup_oss_fuzz(oss_fuzz_dir: Path) -> None:
    sentinel = oss_fuzz_dir / ".fuzzygan_setup"
    if sentinel.exists():
        return
    scripts_dir = oss_fuzz_dir / "infra"
    build_root = oss_fuzz_dir / "build"
    for subdir in ("out", "work", "logs", "temp"):
        (build_root / subdir).mkdir(parents=True, exist_ok=True)
    python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Python interpreter not found for OSS-Fuzz setup")
    requirements = scripts_dir / "cifuzz" / "requirements.txt"
    if requirements.exists():
        run_cmd([python, "-m", "pip", "install", "-r", requirements])
    sentinel.write_text("initialized\n", encoding="utf-8")


def build_library(oss_fuzz_dir: Path, library_name: str) -> None:
    project_dir = oss_fuzz_dir / "projects" / library_name
    build_script = project_dir / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"build.sh not found for {library_name} in {project_dir}")
    build_script.chmod(build_script.stat().st_mode | 0o111)
    env = os.environ.copy()
    env["OUT"] = str(get_build_out_dir(oss_fuzz_dir, library_name))
    run_cmd(["bash", build_script], cwd=project_dir, env=env)


def load_analysis_summary(out_dir: Path, library_name: str) -> dict:
    summary_path = out_dir / library_name / "analysis_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"analysis_summary.json not found for {library_name}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compile_fuzzer(
    harness_path: Path,
    binary_path: Path,
    include_dir: Path,
    library_out_dir: Path,
    library_name: str,
) -> None:
    cmd = [
        "clang",
        "-fsanitize=fuzzer,address",
        "-I",
        str(include_dir),
        "-L",
        str(library_out_dir),
        f"-l{library_name}",
        str(harness_path),
        "-o",
        str(binary_path),
    ]
    run_cmd(cmd)


def run_fuzzing(out_dir: Path, library_name: str, oss_fuzz_dir: Path, storage, run_id: int, db_path: Path) -> None:
    summary = load_analysis_summary(out_dir, library_name)
    project_dir = oss_fuzz_dir / "projects" / library_name
    library_out_dir = get_build_out_dir(oss_fuzz_dir, library_name)
    ensure_build_dirs(out_dir / library_name, library_out_dir)

    for func in summary.get("functions", []):
        if not func.get("worth_fuzzing") or not func.get("harness_path") or not func.get("seed_dir"):
            continue
        harness_path = Path(func["harness_path"])
        binary_path = library_out_dir / f"{func['name']}_fuzzer"
        compile_attempts = 3
        for attempt in range(compile_attempts):
            try:
                compile_fuzzer(harness_path, binary_path, project_dir, library_out_dir, library_name)
                storage.record_event(run_id, "compile_success", f"{func['name']} attempt {attempt + 1}")
                break
            except subprocess.CalledProcessError:
                if attempt == compile_attempts - 1:
                    LOG.error("Failed to compile %s after %s attempts", func["name"], compile_attempts)
                    binary_path = None
                    storage.record_event(run_id, "compile_failed", func["name"])
                else:
                    LOG.info("Retrying compilation for %s", func["name"])
                    storage.record_event(run_id, "compile_retry", f"{func['name']} attempt {attempt + 1}")
        if not binary_path or not binary_path.exists():
            continue
        corpus_dir = Path(func["seed_dir"])
        run_cmd([binary_path, corpus_dir], timeout=300)
        storage.record_event(run_id, "fuzz_invocation", f"{func['name']} with corpus {corpus_dir}")
        diagnostics = intercept(oss_fuzz_dir, db_path, library_name, binary_path, corpus_dir)
        storage.record_event(run_id, "interceptor", json.dumps(diagnostics))


def list_libraries(oss_fuzz_dir: Path) -> List[str]:
    projects_dir = oss_fuzz_dir / "projects"
    if not projects_dir.exists():
        LOG.error("OSS-Fuzz projects directory not found: %s", projects_dir)
        return []
    return sorted(d.name for d in projects_dir.iterdir() if d.is_dir() and (d / "build.sh").exists())


def list_analyzed(out_dir: Path) -> List[str]:
    if not out_dir.exists():
        return []
    return sorted(d.name for d in out_dir.iterdir() if (d / "analysis_summary.json").exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FuzzyGan CLI")
    parser.add_argument("command", choices=["analyze", "fuzz", "list-libraries", "list-analyzed"])
    parser.add_argument("--library")
    parser.add_argument("--header-paths")
    parser.add_argument("--exclude-prefixes")
    parser.add_argument("--skip-functions")
    parser.add_argument("--out-dir", default="fuzz_out")
    parser.add_argument("--oss-fuzz-dir", required=True)
    parser.add_argument("--db-path", default="fuzz_out/fuzzygan.db")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oss_fuzz_dir = Path(args.oss_fuzz_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    db_path = Path(args.db_path).expanduser()

    if args.command == "list-libraries":
        print("Available libraries:", ", ".join(list_libraries(oss_fuzz_dir)) or "None")
        return

    if args.command == "list-analyzed":
        print("Analyzed libraries:", ", ".join(list_analyzed(out_dir)) or "None")
        return

    if not args.library:
        raise SystemExit("Library name required for analyze and fuzz commands")

    setup_oss_fuzz(oss_fuzz_dir)
    config = LibraryConfig.from_args(args, oss_fuzz_dir)

    if args.command == "analyze":
        if not config.header_paths:
            raise SystemExit("Header paths required for analyze command")
        storage = open_storage(db_path)
        run_id = storage.start_run("analyze", config.name, metadata={"out_dir": str(out_dir)})
        try:
            build_library(oss_fuzz_dir, config.name)
            from preanalyze import analyze_library

            analyze_library(
                {
                    "name": config.name,
                    "header_paths": [str(p) for p in config.header_paths],
                    "src_dir": str(config.src_dir),
                    "exclude_prefixes": config.exclude_prefixes,
                    "skip_functions": config.skip_functions,
                },
                str(out_dir),
                storage=storage,
                run_id=run_id,
            )
            storage.finish_run(run_id, "completed")
        except Exception as exc:  # noqa: BLE001
            storage.finish_run(run_id, "failed", {"error": str(exc)})
            storage.close()
            raise
        storage.close()
        return

    if args.command == "fuzz":
        storage = open_storage(db_path)
        run_id = storage.start_run("fuzz", config.name, metadata={"out_dir": str(out_dir)})
        try:
            run_fuzzing(out_dir, config.name, oss_fuzz_dir, storage, run_id, db_path)
            storage.finish_run(run_id, "completed")
        except Exception as exc:  # noqa: BLE001
            storage.finish_run(run_id, "failed", {"error": str(exc)})
            storage.close()
            raise
        storage.close()


if __name__ == "__main__":
    main()
