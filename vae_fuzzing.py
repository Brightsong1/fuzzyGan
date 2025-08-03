import argparse
import json
import torch
import torch.optim as optim
from pathlib import Path
import logging
from vae_model import VAE, vae_loss
from corpus_manager import load_corpus, clean_corpus_dir, save_and_log_corpus
from fuzzer_runner import run_fuzzer, compute_coverage_loss

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
EPOCHS, FUZZ_TIME = 150, 5

def load_summary(summary_file):
    with summary_file.open('r') as f:
        summary = json.load(f)
    worth_fuzzing = [f['name'] for f in summary['functions'] if f['worth_fuzzing']]
    return worth_fuzzing, len(summary['functions']), {f['name']: f for f in summary['functions'] if f['worth_fuzzing'] and f['seed_dir']}

def main():
    parser = argparse.ArgumentParser(description="VAE-based fuzzing")
    parser.add_argument("--function", required=True, help="Function to fuzz")
    parser.add_argument("--library", required=True, help="Library name")
    parser.add_argument("--out-dir", default="fuzz_out", help="Output directory")
    parser.add_argument("--oss-fuzz-dir", help="Path to OSS-Fuzz repository")
    args = parser.parse_args()

    CORPUS_DIR = Path(args.out_dir) / args.library
    OUT_DIR = Path(args.oss_fuzz_dir or "~/Desktop/trspo/nsw/oss-fuzz") / "build" / "out" / args.library
    OUT_DIR = OUT_DIR.expanduser()

    worth_fuzzing, total_functions, functions = load_summary(CORPUS_DIR / "analysis_summary.json")
    if args.function not in functions:
        logging.error(f"Function {args.function} not found or not worth fuzzing")
        return
    func_info = functions[args.function]
    fuzzer_path = OUT_DIR / f"{args.function}_fuzzer"
    if not fuzzer_path.exists():
        logging.error(f"Fuzzer {fuzzer_path} not found")
        return
    corpus_dst = CORPUS_DIR / f"corpus_{args.function}"
    corpus_backup = corpus_dst / "backup"
    if not corpus_backup.exists():
        corpus_backup.mkdir(exist_ok=True)
        for seed in glob.glob(str(Path(func_info['seed_dir']) / "*.bin")):
            shutil.copy(seed, corpus_backup / Path(seed).name)
        logging.info(f"Created corpus backup at {corpus_backup}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    max_edges = 1
    for epoch in range(EPOCHS):
        covered_funcs, covered_edges = run_fuzzer(f"{args.function}_fuzzer", corpus_dst, FUZZ_TIME)
        clean_corpus_dir(corpus_dst)
        max_edges = max(max_edges, covered_edges or 1)
        coverage_loss = compute_coverage_loss(covered_funcs, covered_edges, worth_fuzzing, total_functions, max_edges)
        data = load_corpus(corpus_dst).to(device)
        if len(data) == 0:
            for seed in glob.glob(str(corpus_backup / "*.bin")):
                shutil.copy(seed, corpus_dst / Path(seed).name)
            data = load_corpus(corpus_dst).to(device)
            if len(data) == 0:
                logging.warning(f"Empty corpus for {args.function}, skipping")
                continue
        optimizer.zero_grad()
        recon_data, mu, logvar = vae(data)
        loss = vae_loss(recon_data, data, mu, logvar, coverage_loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z = torch.randn(len(data), vae.latent_dim).to(device)
            new_data = vae.decoder(z).cpu()
            save_and_log_corpus(new_data, corpus_dst, epoch + 1)
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}: Coverage loss = {coverage_loss.item():.4f}, VAE loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()