import re
import json
import time
from pathlib import Path
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def extract_functions_from_header(header_path: str, exclude_prefixes=None) -> list:
    header_path = Path(header_path)
    if not header_path.is_file():
        logging.error(f"Header file not found: {header_path}")
        return []
    with header_path.open('r') as f:
        content = f.read()
    func_re = re.compile(r'(?:/\*.*?\*/\s*)?(?:#if\s+\w+\s+)?(\w[\w\s*]*?)\s+(\w+)\s*\(([^)]*)\)\s*;', re.DOTALL | re.MULTILINE)
    functions = []
    for match in func_re.finditer(content):
        return_type, func_name, params = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        if exclude_prefixes and any(func_name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', func_name) or not re.match(r'^[\w\s*]+$', return_type):
            continue
        functions.append({'name': func_name, 'signature': f"{return_type} {func_name}({params});"})
    logging.info(f"Extracted {len(functions)} functions from {header_path}")
    return functions

def extract_function_definition(func_name: str, src_dir: str) -> str:
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        return "Implementation not found."
    func_re = re.compile(r'(?:^|\n)\s*([\w\s*]+)\s+\b' + re.escape(func_name) + r'\b\s*\(([^)]*)\)\s*', re.MULTILINE)
    for c_file in src_dir.rglob("*.c"):  # Changed to rglob for recursive search
        with c_file.open('r') as f:
            content = f.read()
        match = func_re.search(content)
        if match:
            start_idx = match.start()
            brace_count, i = 0, content.find('{', start_idx)
            if i == -1:
                continue
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return f"{match.group(1).strip()} {func_name}({match.group(2).strip()}) {content[start_idx:i+1]}"
                i += 1
    return "Implementation not found."

def query_model(prompt: str, func_name: str, library_name: str) -> dict:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-pro')
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            content = response.text.strip()
            content = re.sub(r'```json\n|\n```', '', content)
            try:
                data = json.loads(content)
                if not all(k in data for k in ["worth_fuzzing", "filename", "harness", "seeds", "explain"]):
                    raise ValueError("Missing required fields")
                if data["worth_fuzzing"] and not data["harness"].startswith(f"#include <{library_name}.h>"):
                    raise ValueError("Invalid harness header")
                if data["worth_fuzzing"] and not all(re.match(r'^[0-9a-fA-F]+$', s) for s in data["seeds"]):
                    raise ValueError("Invalid seed format")
                return data
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON for {func_name}: {content[:200]}")
                harness = re.sub(r'^```[a-z]*\n|\n```$', '', content, flags=re.MULTILINE)
                seeds = re.findall(r'"([0-9a-fA-F]+)"', content)
                return {"worth_fuzzing": bool(harness), "filename": f"{func_name}.c", "harness": harness, "seeds": seeds, "explain": "Fallback parsing"}
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {func_name}: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                return {"worth_fuzzing": False, "explain": f"Failed after 3 attempts: {e}"}

def save_artifacts(out_dir: str, library_name: str, func_name: str, response: dict) -> dict:
    func_dir = Path(out_dir) / library_name / func_name
    func_dir.mkdir(parents=True, exist_ok=True)
    if not response.get("worth_fuzzing", False):
        return {}
    harness_file = func_dir / f"{func_name}.c"
    harness_file.write_text(response["harness"])
    seed_dir = func_dir / func_name
    seed_dir.mkdir(exist_ok=True)
    seeds = response.get("seeds", [])
    for idx, seed_hex in enumerate(seeds):
        try:
            seed_bytes = bytes.fromhex(seed_hex)
            (seed_dir / f"seed_{idx:04d}.bin").write_bytes(seed_bytes)
        except ValueError:
            logging.warning(f"Invalid seed for {func_name}: {seed_hex}")
    return {"harness_path": str(harness_file), "seed_dir": str(seed_dir), "num_seeds": len(seeds)}

def analyze_library(library_config: dict, out_dir: str) -> dict:
    library_name = library_config["name"]
    out_dir = Path(out_dir) / library_name
    out_dir.mkdir(parents=True, exist_ok=True)
    functions = []
    for header_path in library_config["header_paths"]:
        functions.extend(extract_functions_from_header(header_path, library_config.get("exclude_prefixes", [])))
    with open("prompt.txt", "r") as f:
        prompt_template = f.read()
    analysis_results = []
    for func in functions:
        if func["name"] in library_config.get("skip_functions", []):
            continue
        definition = extract_function_definition(func["name"], library_config["src_dir"])
        prompt = prompt_template.format(library_name=library_name, func_name=func["name"], signature=func["signature"], implementation=definition)
        response = query_model(prompt, func["name"], library_name)
        artifacts = save_artifacts(out_dir, library_name, func["name"], response)
        analysis_results.append({"name": func["name"], "worth_fuzzing": response.get("worth_fuzzing", False), **artifacts})
    summary_path = out_dir / "analysis_summary.json"
    summary = {"library": library_name, "functions": analysis_results}
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)
    return summary