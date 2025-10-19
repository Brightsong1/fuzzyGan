import re
import select
import signal
import subprocess
import time
from pathlib import Path

import torch

def run_fuzzer(fuzzer_path, corpus_dir, timeout=10):
    fuzzer_path = Path(fuzzer_path)
    cmd = [f"./{fuzzer_path.name}", str(Path(corpus_dir).resolve()), "-max_len=1048576", "-timeout=10", "-dump_coverage=1", "-print_coverage=1"]
    stdout_lines, stderr_lines = [], []
    process = subprocess.Popen(cmd, cwd=str(fuzzer_path.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_time = time.time()
    end_time = start_time + timeout
    while process.poll() is None and time.time() < end_time:
        rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)
        for stream in rlist:
            line = stream.readline().strip()
            if stream == process.stdout:
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
    else:
        stdout, stderr = process.communicate(timeout=5)
    stdout_lines.extend(stdout.splitlines())
    stderr_lines.extend(stderr.splitlines())
    covered_funcs = set()
    observed_funcs = set()
    covered_edges = 0
    covered_re = re.compile(r"^COVERED_FUNC:\s+hits:\s+\d+\s+edges:\s+(\d+)/(\d+)\s+(.*)$")
    uncovered_re = re.compile(r"^UNCOVERED_FUNC:\s+hits:\s+\d+\s+edges:\s+\d+/\d+\s+(.*)$")
    for line in stderr_lines:
        if match := covered_re.match(line):
            covered_edges += int(match.group(1))
            label = match.group(3).strip()
            covered_funcs.add(label)
            observed_funcs.add(label)
        elif match := uncovered_re.match(line):
            observed_funcs.add(match.group(1).strip())
    return covered_funcs, covered_edges, observed_funcs

def compute_coverage_loss(covered_funcs, observed_funcs, covered_edges, worth_fuzzing, total_functions, max_edges):
    if worth_fuzzing:
        targets = set(worth_fuzzing)
        func_coverage_ratio = len(targets & covered_funcs) / max(1, len(targets))
        if func_coverage_ratio == 0 and observed_funcs:
            func_coverage_ratio = len(covered_funcs) / max(1, len(observed_funcs))
    else:
        func_coverage_ratio = len(covered_funcs) / max(1, len(observed_funcs) or total_functions or len(covered_funcs))
    code_coverage_ratio = covered_edges / max(1, max_edges)
    return (
        torch.tensor(0.7 * (1 - func_coverage_ratio) + 0.3 * (1 - code_coverage_ratio), requires_grad=True),
        func_coverage_ratio,
        code_coverage_ratio,
    )
