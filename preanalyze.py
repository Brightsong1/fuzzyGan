import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from storage import Storage

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("fuzzygan.preanalyze")

MODEL_RETRIES = 3
RETRY_DELAY = 5
REGEN_ATTEMPTS = 3
MIN_SEEDS = 3
SOURCE_GLOBS = ("*.c", "*.h", "*.txt")
MAX_BYTES_PER_TARGET = 16000
MAX_FILES_PER_TARGET = 12


@dataclass
class KernelTarget:
    kernel_name: str
    target_path: Path
    code_excerpt: str


def escape_template_value(value: str) -> str:
    return value.replace("$", "$$")


def configure_model() -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")


def call_model(model: genai.GenerativeModel, prompt: str) -> str:
    for attempt in range(1, MODEL_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            LOG.error("Model invocation failed (attempt %d/%d): %s", attempt, MODEL_RETRIES, exc)
            if attempt == MODEL_RETRIES:
                raise
            time.sleep(RETRY_DELAY)
    return ""


def clean_model_output(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"`{3}$", "", cleaned)
    return cleaned.strip()


def parse_model_response(raw_text: str) -> Tuple[Optional[Dict], List[str]]:
    if not raw_text:
        return None, ["Model returned an empty response."]
    cleaned = clean_model_output(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, ["Model response was not valid JSON."]
    required = {"worth_fuzzing", "name", "subsystem", "syz_program", "seeds", "explain"}
    missing = sorted(required - data.keys())
    if missing:
        return None, [f"Missing keys in response: {', '.join(missing)}."]
    return data, []


def validate_response(response: Dict) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    worth = response.get("worth_fuzzing")
    if not isinstance(worth, bool):
        issues.append('"worth_fuzzing" must be a boolean.')
    if worth:
        if not isinstance(response.get("name"), str) or not response["name"].strip():
            issues.append('"name" must be a non-empty string.')
        if not isinstance(response.get("subsystem"), str) or not response["subsystem"].strip():
            issues.append('"subsystem" must be a non-empty string.')
        syz_prog = response.get("syz_program", "")
        if not isinstance(syz_prog, str) or not syz_prog.strip():
            issues.append('"syz_program" must be a non-empty syzlang program.')
        seeds = response.get("seeds", [])
        if not isinstance(seeds, list) or len(seeds) < MIN_SEEDS:
            issues.append(f'Provide at least {MIN_SEEDS} seed syz programs in "seeds".')
        else:
            for seed in seeds:
                if not isinstance(seed, str) or not seed.strip():
                    issues.append("Seed programs must be non-empty strings.")
                    break
    return len(issues) == 0, issues


def build_feedback(issues: List[str]) -> str:
    bullet_lines = [f"- {issue}" for issue in issues] or ["- No specific issues captured."]
    return (
        "Previous attempt issues:\n"
        f"{'\n'.join(bullet_lines)}\n"
        "Regenerate a complete JSON object that resolves every issue above. "
        "Do not reuse the earlier program verbatim; vary syscalls/arguments."
    )


def generate_with_regeneration(
    model: genai.GenerativeModel,
    template: Template,
    prompt_context: Dict[str, str],
    storage: Optional[Storage],
    run_id: Optional[int],
) -> Dict:
    extra_guidance = "No additional guidance; follow the instructions carefully."
    last_issues: List[str] = []
    for attempt in range(1, REGEN_ATTEMPTS + 1):
        context = dict(prompt_context)
        context["extra_guidance"] = escape_template_value(extra_guidance)
        prompt = template.safe_substitute(context)
        raw_text = call_model(model, prompt)
        response, parse_issues = parse_model_response(raw_text)
        if parse_issues:
            last_issues = parse_issues
            extra_guidance = build_feedback(parse_issues)
            if storage and run_id:
                storage.record_event(run_id, "llm_parse_error", "; ".join(parse_issues))
            LOG.warning("Model response parsing failed (attempt %d/%d)", attempt, REGEN_ATTEMPTS)
            continue
        valid, validation_issues = validate_response(response or {})
        if not valid:
            last_issues = validation_issues
            extra_guidance = build_feedback(validation_issues)
            if storage and run_id:
                storage.record_event(run_id, "llm_validation_error", "; ".join(validation_issues))
            LOG.warning("Validation failed (attempt %d/%d): %s", attempt, REGEN_ATTEMPTS, "; ".join(validation_issues))
            continue
        if storage and run_id:
            storage.record_event(run_id, "llm_success", response["name"])
        return response  # type: ignore[return-value]
    return {
        "worth_fuzzing": False,
        "name": "skip",
        "subsystem": "",
        "syz_program": "",
        "seeds": [],
        "explain": f"Failed to generate a valid program after {REGEN_ATTEMPTS} attempts: {'; '.join(last_issues)}",
    }


def load_prompt() -> Template:
    template_path = Path("prompt.txt")
    content = template_path.read_text(encoding="utf-8")
    return Template(content)


def collect_code_excerpt(target_dir: Path, max_files: int, max_bytes: int) -> str:
    snippets: List[str] = []
    total_budget = max_bytes
    files: List[Path] = []
    for glob in SOURCE_GLOBS:
        files.extend(sorted(target_dir.rglob(glob)))
    files = files[:max_files]
    per_file_budget = max(512, total_budget // max(1, len(files)))
    for file in files:
        try:
            raw = file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet = raw[:per_file_budget]
        rel = file.relative_to(target_dir)
        snippets.append(f"// {rel}\n{snippet}")
        if sum(len(s) for s in snippets) >= total_budget:
            break
    joined = "\n\n".join(snippets)
    return joined[:total_budget]


def enumerate_targets(kernel_src: Path, focus_dirs: List[str], max_files: int, max_bytes: int) -> List[KernelTarget]:
    targets: List[KernelTarget] = []
    for sub_path in focus_dirs:
        target_dir = (kernel_src / sub_path).resolve()
        if not target_dir.exists():
            LOG.warning("Target path not found, skipping: %s", target_dir)
            continue
        excerpt = collect_code_excerpt(target_dir, max_files, max_bytes)
        targets.append(KernelTarget(kernel_src.name, target_dir, excerpt))
    return targets


def save_artifacts(out_dir: Path, kernel_name: str, target_path: Path, response: Dict) -> Dict:
    target_root = out_dir / kernel_name
    programs_dir = target_root / "programs"
    seeds_dir = target_root / "seeds" / response["name"]
    programs_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    program_path = programs_dir / f"{response['name']}.syzprog"
    if response.get("syz_program"):
        program_path.write_text(response["syz_program"], encoding="utf-8")

    seed_entries = []
    for index, seed in enumerate(response.get("seeds", [])):
        seed_path = seeds_dir / f"seed_{index:04d}.syzprog"
        seed_path.write_text(seed, encoding="utf-8")
        seed_entries.append(str(seed_path))

    return {
        "name": response.get("name", ""),
        "subsystem": response.get("subsystem", ""),
        "worth_fuzzing": response.get("worth_fuzzing", False),
        "program_path": str(program_path),
        "seed_dir": str(seeds_dir),
        "num_seeds": len(seed_entries),
    }


def analyze_kernel(
    kernel_config: Dict,
    out_dir: str,
    storage: Optional[Storage] = None,
    run_id: Optional[int] = None,
) -> Dict:
    kernel_name = kernel_config["name"]
    kernel_src = Path(kernel_config["kernel_src"]).expanduser().resolve()
    focus_dirs = kernel_config.get("focus_dirs") or ["drivers", "fs", "net"]
    max_files = int(kernel_config.get("max_files", MAX_FILES_PER_TARGET))
    max_bytes = int(kernel_config.get("max_bytes", MAX_BYTES_PER_TARGET))

    destination = Path(out_dir) / kernel_name
    destination.mkdir(parents=True, exist_ok=True)

    template = load_prompt()
    model = configure_model()

    targets = enumerate_targets(kernel_src, focus_dirs, max_files, max_bytes)
    summary_results = []
    for target in targets:
        prompt_context = {
            "kernel_name": escape_template_value(kernel_name),
            "target_path": escape_template_value(str(target.target_path.relative_to(kernel_src))),
            "code_excerpt": escape_template_value(target.code_excerpt),
        }
        response = generate_with_regeneration(model, template, prompt_context, storage, run_id)
        artifacts = save_artifacts(destination, kernel_name, target.target_path, response)
        summary_results.append({**artifacts, "target_path": str(target.target_path)})

    summary = {"kernel": kernel_name, "targets": summary_results}
    summary_path = destination / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if storage and run_id:
        storage.record_event(run_id, "analysis_summary_written", str(summary_path))
    return summary
