import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from storage import Storage, open_storage


class Interceptor:
    def __init__(self, oss_fuzz_dir: Path, storage: Storage):
        self.oss_fuzz_dir = oss_fuzz_dir
        self.storage = storage

    def run_diagnostics(self, library: str, fuzzer_binary: Path, corpus_dir: Path, timeout: int = 60) -> Dict[str, object]:
        cmd = [str(fuzzer_binary), str(corpus_dir), "-runs=0", "-print_final_stats=1"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            return {"error": f"diagnostics timeout after {timeout}s", "stdout": exc.stdout, "stderr": exc.stderr}

    def compare_with_history(self, library: str) -> Dict[str, object]:
        runs = list(self.storage.summarize_runs())
        relevant = [run for run in runs if run["library"] == library]
        if len(relevant) < 2:
            return {"message": "insufficient historical runs"}
        latest, previous = relevant[0], relevant[1]
        delta = (latest["finished_at"] or time.time()) - (previous["finished_at"] or time.time())
        return {
            "latest_status": latest["status"],
            "previous_status": previous["status"],
            "time_between_runs": delta,
        }


def intercept(oss_fuzz_dir: Path, db_path: Path, library: str, fuzzer_binary: Path, corpus_dir: Path, timeout: int = 60) -> Dict[str, object]:
    storage = open_storage(db_path)
    interceptor = Interceptor(oss_fuzz_dir, storage)
    diagnostics = interceptor.run_diagnostics(library, fuzzer_binary, corpus_dir, timeout=timeout)
    comparison = interceptor.compare_with_history(library)
    storage.close()
    return {"diagnostics": diagnostics, "comparison": comparison}

