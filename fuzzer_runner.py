import subprocess
import select
import time
import re
import logging

def run_fuzzer(fuzzer, corpus_dir, timeout=10):
    cmd = [f"./{fuzzer}", str(corpus_dir), "-max_len=1048576", "-timeout=10", "-dump_coverage=1", "-print_coverage=1"]
    stdout_lines, stderr_lines = [], []
    process = subprocess.Popen(cmd, cwd=corpus_dir.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_time = time.time()
    while process.poll() is None and (time.time() - start_time) < timeout:
        rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)
        for stream in rlist:
            line = stream.readline().strip()
            if stream == process.stdout:
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
    stdout, stderr = process.communicate(timeout=5)
    stdout_lines.extend(stdout.splitlines())
    stderr_lines.extend(stderr.splitlines())
    covered_funcs = [parts[5] for line in stderr_lines if line.startswith("COVERED_FUNC:") and len(parts := line.split()) >= 6 and "/" not in parts[5]]
    covered_edges, total_edges = 0, 0
    for line in stderr_lines:
        if match := re.search(r"edges: (\d+)/(\d+)", line):
            covered_edges += int(match.group(1))
            total_edges += int(match.group(2))
    return covered_funcs, covered_edges

def compute_coverage_loss(covered_funcs, covered_edges, worth_fuzzing, total_functions, max_edges):
    func_coverage_ratio = len(set(covered_funcs) & set(worth_fuzzing)) / total_functions if total_functions else 0
    code_coverage_ratio = covered_edges / max_edges if max_edges else 0
    return torch.tensor(0.7 * (1 - func_coverage_ratio) + 0.3 * (1 - code_coverage_ratio), requires_grad=True)