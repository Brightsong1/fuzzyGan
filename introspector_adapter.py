import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

LOG = logging.getLogger("fuzzygan.introspector")


def _candidate_paths(project: str,
                     oss_fuzz_dir: Path,
                     extras: Iterable[Path] = ()) -> Iterable[Path]:
    env_out = os.getenv("FUZZ_INTROSPECTOR_OUTPUT")
    if env_out:
        env_root = Path(env_out)
        yield env_root / project / "inspector-report" / "summary.json"
        yield env_root / "inspector-report" / "summary.json"
        yield env_root / "summary.json"

    build_out = Path(oss_fuzz_dir) / "build" / "out" / project
    yield build_out / "inspector-report" / "summary.json"
    yield build_out / "inspector" / "summary.json"

    for extra in extras:
        extra_path = Path(extra)
        if extra_path.is_dir():
            yield extra_path / "summary.json"
        yield extra_path


def _load_summary(project: str,
                  oss_fuzz_dir: Path,
                  extras: Iterable[Path] = ()) -> Optional[Dict]:
    for candidate in _candidate_paths(project, oss_fuzz_dir, extras):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOG.warning("Failed to decode Fuzz Introspector summary %s: %s",
                            candidate, exc)
    return None


def _simplify_summary(summary: Dict) -> Dict:
    result: Dict[str, Dict] = {"overall": {}, "fuzzers": {}, "optimal_targets": []}
    merged = summary.get("MergedProjectProfile", {})
    stats = merged.get("stats", {})
    result["overall"] = {
        "functions_total": stats.get("function-count"),
        "functions_reached": stats.get("functions-reached"),
        "functions_unreached": stats.get("functions-unreached"),
        "basic_blocks_total": stats.get("total-basic-blocks"),
        "unreached_complexity_pct": stats.get("unreached-complexity-percentage"),
    }

    analyses = summary.get("analyses") or {}
    optimal = analyses.get("OptimalTargets") or {}
    if isinstance(optimal, dict):
        # The analysis payload is a list under the key 'optimal-targets'
        targets = optimal.get("optimal-targets") or []
        if isinstance(targets, list):
            result["optimal_targets"] = targets[:20]

    for name, payload in summary.items():
        if name in {"MergedProjectProfile", "analyses"}:
            continue
        if not isinstance(payload, dict):
            continue
        stats = payload.get("stats", {})
        coverage_stats = payload.get("coverage-blocker-stats", {})
        result["fuzzers"][name] = {
            "basic_blocks": stats.get("total-basic-blocks"),
            "functions_total": stats.get("function-count"),
            "functions_reached": stats.get("functions-reached"),
            "cov_reach_proportion": coverage_stats.get("cov-reach-proportion"),
            "reachability_score": coverage_stats.get("reachability-score"),
        }
    return result


def collect_summary(
    project: str,
    oss_fuzz_dir: Path,
    extras: Iterable[Path] = (),
) -> Optional[Dict]:
    summary = _load_summary(project, oss_fuzz_dir, extras)
    if summary is None:
        env_hint = os.getenv("FUZZ_INTROSPECTOR_OUTPUT")
        LOG.warning(
            "Fuzz Introspector summary not found for %s. "
            "Ensure the project is built with FUZZ_INTROSPECTOR=1 and "
            "post-processing has been executed. Checked FUZZ_INTROSPECTOR_OUTPUT=%s",
            project,
            env_hint,
        )
        return None
    return _simplify_summary(summary)


def ensure_python_path(root: Path) -> None:
    introspector_src = root / "tools" / "fuzz-introspector" / "src"
    if introspector_src.is_dir():
        sys.path.append(str(introspector_src))
