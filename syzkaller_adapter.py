import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LOG = logging.getLogger("fuzzygan.syzkaller")


def load_manager_config(config_path: Path) -> Dict:
    """Load a syzkaller manager config (manager.cfg)."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "workdir" not in cfg:
        raise ValueError("manager config missing required field 'workdir'")
    return cfg


def workdir_from_config(cfg: Dict) -> Path:
    return Path(cfg["workdir"]).expanduser()


def corpus_root(cfg: Dict) -> Path:
    return workdir_from_config(cfg) / "corpus"


def stage_corpus(sources: Iterable[Path], cfg: Dict, fuzzer_prefix: str = "fuzzer-0") -> List[Path]:
    """Copy generated syz programs/seeds into the syzkaller corpus directory.

    syzkaller expects per-fuzzer directories under workdir/corpus/*. We drop files
    into workdir/corpus/{fuzzer_prefix}/ while preserving filenames.
    """
    corpus_dir = corpus_root(cfg) / fuzzer_prefix
    corpus_dir.mkdir(parents=True, exist_ok=True)
    staged: List[Path] = []
    for source in sources:
        source = source.expanduser()
        if not source.exists():
            LOG.warning("Source not found for staging: %s", source)
            continue
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            target = corpus_dir / path.name
            shutil.copy2(path, target)
            staged.append(target)
    LOG.info("Staged %d files into %s", len(staged), corpus_dir)
    return staged


def _parse_json_line(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def read_latest_stats(stats_path: Path) -> Dict:
    """Parse the most recent JSON line from fuzzer.stats (syzkaller writes NDJSON)."""
    stats_path = stats_path.expanduser()
    if not stats_path.exists():
        return {}
    lines = stats_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        parsed = _parse_json_line(line.strip())
        if parsed is not None:
            return parsed
    return {}


def scan_crashes(workdir: Path) -> List[Path]:
    crash_dir = workdir.expanduser() / "crashes"
    if not crash_dir.exists():
        return []
    return sorted(p for p in crash_dir.iterdir() if p.is_dir())


def summarize_feedback(stats_path: Path, crash_dir: Path) -> Tuple[int, List[str], Dict]:
    """Return (covered_edges, covered_syscalls, raw_stats) derived from syzkaller outputs."""
    stats = read_latest_stats(stats_path)
    covered_edges = int(stats.get("cover", 0) or 0)
    syscalls = stats.get("syscalls") or []
    covered_syscalls = [str(s) for s in syscalls] if isinstance(syscalls, Iterable) else []
    crashes = scan_crashes(crash_dir)
    if crashes:
        stats["crash_dirs"] = [str(p) for p in crashes]
        stats["crash_count"] = len(crashes)
    return covered_edges, covered_syscalls, stats
