from pathlib import Path
from typing import Iterable, Set, Tuple

import torch

from syzkaller_adapter import summarize_feedback


def run_fuzzer(stats_path: Path, crash_dir: Path) -> Tuple[Set[str], int, Set[str]]:
    """Read syzkaller stats/crash data and return coverage metrics.

    covered_funcs: set of syscalls reported by syzkaller
    covered_edges: edge coverage from stats
    observed_funcs: same as covered_funcs (syzkaller does not report unreached syscalls here)
    """
    covered_edges, covered_syscalls, _ = summarize_feedback(stats_path, crash_dir)
    covered_funcs = set(covered_syscalls)
    return covered_funcs, covered_edges, set(covered_syscalls)


def compute_coverage_loss(
    covered_funcs: Iterable[str],
    observed_funcs: Iterable[str],
    covered_edges: int,
    worth_fuzzing: Iterable[str],
    total_functions: int,
    max_edges: int,
):
    targets = set(worth_fuzzing or [])
    covered_funcs_set = set(covered_funcs)
    observed_funcs_set = set(observed_funcs) or covered_funcs_set
    if targets:
        func_coverage_ratio = len(targets & covered_funcs_set) / max(1, len(targets))
        if func_coverage_ratio == 0 and observed_funcs_set:
            func_coverage_ratio = len(covered_funcs_set) / max(1, len(observed_funcs_set))
    else:
        func_coverage_ratio = len(covered_funcs_set) / max(1, len(observed_funcs_set) or total_functions or len(covered_funcs_set))
    code_coverage_ratio = covered_edges / max(1, max_edges)
    return (
        torch.tensor(0.7 * (1 - func_coverage_ratio) + 0.3 * (1 - code_coverage_ratio), requires_grad=True),
        func_coverage_ratio,
        code_coverage_ratio,
    )
