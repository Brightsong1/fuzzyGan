import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
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
SOURCE_GLOBS = ("*.c", "*.cc", "*.cpp", "*.cxx")


def escape_template_value(value: str) -> str:
    return value.replace("$", "$$")


def extract_functions_from_header(header_path: str, exclude_prefixes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    header = Path(header_path)
    if not header.is_file():
        LOG.error("Header file not found: %s", header)
        return []
    content = header.read_text(encoding="utf-8", errors="ignore")
    prototype = re.compile(
        r"""
        (?:
            /\*.*?\*/\s* |
            //.*?$ |
            \#ifdef.*?\#endif\s* |
            \#if.*?\#endif\s*
        )*
        ([\w\s*\(\)\[\],]+?)\s+
        (\w+)\s*\(([^;{}]*)\)\s*;
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )
    functions: List[Dict[str, str]] = []
    for match in prototype.finditer(content):
        return_type = " ".join(match.group(1).split())
        func_name = match.group(2)
        params = match.group(3).strip()
        if exclude_prefixes and any(func_name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", func_name):
            continue
        functions.append({"name": func_name, "signature": f"{return_type} {func_name}({params});"})
    LOG.info("Extracted %d candidate functions from %s", len(functions), header)
    return functions


def extract_function_definition(func_name: str, src_dir: str) -> str:
    src_root = Path(src_dir)
    if not src_root.is_dir():
        return "Implementation not found."
    definition_pattern = re.compile(rf"(?:^|\n)\s*([\w\s\*]+)\b{re.escape(func_name)}\s*\(([^)]*)\)\s*{{", re.MULTILINE)
    for glob_pattern in SOURCE_GLOBS:
        for source_file in src_root.rglob(glob_pattern):
            content = source_file.read_text(encoding="utf-8", errors="ignore")
            match = definition_pattern.search(content)
            if not match:
                continue
            start = match.start()
            brace_depth = 0
            body_start = content.find("{", start)
            if body_start == -1:
                continue
            index = body_start
            while index < len(content):
                if content[index] == "{":
                    brace_depth += 1
                elif content[index] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        return content[start : index + 1]
                index += 1
    return "Implementation not found."


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
        except Exception as exc:
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
    required = {"worth_fuzzing", "filename", "harness", "seeds", "explain"}
    missing = sorted(required - data.keys())
    if missing:
        return None, [f"Missing keys in response: {', '.join(missing)}."]
    return data, []


def validate_response(library_name: str, response: Dict) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not isinstance(response.get("worth_fuzzing"), bool):
        issues.append('"worth_fuzzing" must be a boolean.')
    if response.get("worth_fuzzing"):
        harness = response.get("harness", "")
        seeds = response.get("seeds", [])
        if f"#include <{library_name}.h>" not in harness and f'#include "{library_name}.h"' not in harness:
            issues.append("Harness must include the library header.")
        if "LLVMFuzzerTestOneInput" not in harness:
            issues.append("Harness must define LLVMFuzzerTestOneInput.")
        if not isinstance(seeds, list) or len(seeds) < MIN_SEEDS:
            issues.append(f"Provide at least {MIN_SEEDS} hex-encoded seeds.")
        else:
            for raw_seed in seeds:
                if not isinstance(raw_seed, str) or not raw_seed:
                    issues.append("Seeds must be non-empty strings.")
                    break
                if len(raw_seed) % 2 != 0 or not re.fullmatch(r"[0-9a-fA-F]+", raw_seed):
                    issues.append(f"Seed '{raw_seed}' is not valid hexadecimal.")
                    break
                if len(raw_seed) < 4:
                    issues.append("Seeds must include realistic headers and be at least 2 bytes long.")
                    break
        filename = response.get("filename", "")
        if not isinstance(filename, str) or not filename.endswith(".c"):
            issues.append('Filename must be a ".c" source file.')
    return len(issues) == 0, issues


def build_feedback(issues: List[str], compile_errors: Optional[str] = None) -> str:
    bullet_lines = [f"- {issue}" for issue in issues]
    if compile_errors:
        bullet_lines.append(f"- Resolve compiler errors:\n{compile_errors}")
    bullet_list = "\n".join(bullet_lines) if bullet_lines else "- No specific issues captured."
    return (
        "Previous attempt issues:\n"
        f"{bullet_list}\n"
        "Regenerate a complete JSON object that resolves every issue above. "
        "Do not reuse the earlier harness verbatim."
    )


def compile_candidate(harness: str, include_dirs: List[Path]) -> Tuple[bool, Optional[str]]:
    clang = shutil.which("clang") or shutil.which("cc")
    if not clang:
        return True, None
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tmp:
        tmp.write(harness)
        tmp_path = Path(tmp.name)
    cmd = [clang, "-fsyntax-only", str(tmp_path)]
    for include_dir in include_dirs:
        cmd.extend(["-I", str(include_dir)])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_with_regeneration(
    model: genai.GenerativeModel,
    template: Template,
    prompt_context: Dict[str, str],
    library_name: str,
    func_name: str,
    include_dirs: List[Path],
    storage: Optional[Storage],
    run_id: Optional[int],
) -> Dict:
    extra_guidance = "No additional guidance; follow the instructions carefully."
    last_issues: List[str] = []
    compile_errors: Optional[str] = None
    for attempt in range(1, REGEN_ATTEMPTS + 1):
        context = dict(prompt_context)
        context["extra_guidance"] = escape_template_value(extra_guidance)
        prompt = template.safe_substitute(context)
        raw_text = call_model(model, prompt)
        response, parse_issues = parse_model_response(raw_text)
        if parse_issues:
            last_issues = parse_issues
            extra_guidance = build_feedback(parse_issues, compile_errors)
            if storage and run_id:
                storage.record_event(run_id, "llm_parse_error", f"{func_name}: {'; '.join(parse_issues)}")
            LOG.warning("Model response parsing failed for %s (attempt %d/%d)", func_name, attempt, REGEN_ATTEMPTS)
            continue
        valid, validation_issues = validate_response(library_name, response or {})
        if not valid:
            last_issues = validation_issues
            extra_guidance = build_feedback(validation_issues, compile_errors)
            if storage and run_id:
                storage.record_event(run_id, "llm_validation_error", f"{func_name}: {'; '.join(validation_issues)}")
            LOG.warning("Validation failed for %s (attempt %d/%d): %s", func_name, attempt, REGEN_ATTEMPTS, "; ".join(validation_issues))
            continue
        compiles, errors = compile_candidate(response["harness"], include_dirs)
        if compiles:
            if storage and run_id:
                storage.record_event(run_id, "llm_success", func_name)
            return response  # type: ignore[return-value]
        compile_errors = errors or "Unknown compiler failure."
        if storage and run_id:
            storage.record_event(run_id, "llm_compile_error", f"{func_name}: {compile_errors}")
        extra_guidance = build_feedback(["Harness failed to compile."], compile_errors)
    return {
        "worth_fuzzing": False,
        "filename": "",
        "harness": "",
        "seeds": [],
        "explain": f"Failed to generate a valid harness after {REGEN_ATTEMPTS} attempts: {compile_errors or '; '.join(last_issues)}",
    }


def save_artifacts(out_dir: Path, library_name: str, func_name: str, response: Dict) -> Dict:
    func_dir = out_dir / library_name / func_name
    func_dir.mkdir(parents=True, exist_ok=True)
    if not response.get("worth_fuzzing"):
        return {}
    harness_file = func_dir / response["filename"]
    harness_file.write_text(response["harness"], encoding="utf-8")
    seed_dir = func_dir / func_name
    seed_dir.mkdir(exist_ok=True)
    seeds = response.get("seeds", [])
    for index, seed_hex in enumerate(seeds):
        seed_bytes = bytes.fromhex(seed_hex)
        (seed_dir / f"seed_{index:04d}.bin").write_bytes(seed_bytes)
    return {"harness_path": str(harness_file), "seed_dir": str(seed_dir), "num_seeds": len(seeds)}


def load_prompt() -> Template:
    template_path = Path("prompt.txt")
    content = template_path.read_text(encoding="utf-8")
    return Template(content)


def analyze_library(library_config: Dict, out_dir: str, storage: Optional[Storage] = None, run_id: Optional[int] = None) -> Dict:
    library_name = library_config["name"]
    destination = Path(out_dir) / library_name
    destination.mkdir(parents=True, exist_ok=True)
    template = load_prompt()
    model = configure_model()

    include_dirs_set = {Path(library_config["src_dir"])}
    for path in library_config.get("header_paths", []):
        header_path = Path(path)
        include_dirs_set.add(header_path.parent if header_path.is_file() else header_path)
    include_dirs = sorted(include_dirs_set, key=lambda p: str(p))

    functions: List[Dict[str, str]] = []
    for header_path in library_config.get("header_paths", []):
        functions.extend(extract_functions_from_header(header_path, library_config.get("exclude_prefixes", [])))

    summary_results = []
    for func in functions:
        if func["name"] in library_config.get("skip_functions", []):
            continue
        implementation = extract_function_definition(func["name"], library_config["src_dir"])
        prompt_context = {
            "library_name": escape_template_value(library_name),
            "func_name": escape_template_value(func["name"]),
            "signature": escape_template_value(func["signature"]),
            "implementation": escape_template_value(implementation),
        }
        response = generate_with_regeneration(
            model,
            template,
            prompt_context,
            library_name,
            func["name"],
            include_dirs,
            storage,
            run_id,
        )
        artifacts = save_artifacts(destination, library_name, func["name"], response)
        summary_results.append(
            {
                "name": func["name"],
                "worth_fuzzing": response.get("worth_fuzzing", False),
                **artifacts,
            }
        )

    summary = {"library": library_name, "functions": summary_results}
    summary_path = destination / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if storage and run_id:
        storage.record_event(run_id, "analysis_summary_written", str(summary_path))
    return summary
