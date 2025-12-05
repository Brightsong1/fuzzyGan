# FuzzyGan: VAE-guided syzkaller pipeline for Linux kernel fuzzing

## Overview
This branch of FuzzyGan pivots from OSS-Fuzz/LibFuzzer harness generation to Linux kernel fuzzing with syzkaller. The pipeline:
- Pre-analyzes the kernel tree, feeds code slices to an LLM, and emits syzlang programs plus seed corpora.
- Uses a Variational Autoencoder (VAE) to mutate syz programs/corpora based on coverage feedback from syzkaller stats/logs.
- Stores per-epoch metrics and corpus snapshots for later inspection/training.

## Project Structure
- fuzzer.py: CLI for kernel preanalysis (LLM -> syzlang) and corpus staging into syzkaller workdirs.
- preanalyze.py: Kernel slicer + LLM caller that produces syzkaller programs/seeds and `analysis_summary.json`.
- prompt.txt: Template prompt for generating syzlang scripts and seed corpora.
- syzkaller_adapter.py: Helpers to read `manager.cfg`, stage corpora, and parse syzkaller stats/crash data.
- vae_model.py: VAE architecture and loss.
- corpus_manager.py: Corpus load/save utilities (now syzprog-aware).
- vae_fuzzing.py: VAE training loop using syzkaller feedback and mutating syz corpora.
- storage.py, policy.py: Run/event storage and adaptive mutation policy.

## Requirements
- Python 3.8+
- torch, google-generativeai, numpy
- Environment: `GEMINI_API_KEY` for LLM calls
- A local syzkaller checkout (now a submodule at `./syzkaller`) and a built kernel with kcov/KASAN/Config instrumentation.
- A valid `manager.cfg` 

Install dependencies:
```
pip install torch google-generativeai numpy
```

Set API key:
```
export GEMINI_API_KEY="your-api-key"
```

## Usage (syzkaller)
- Preanalyze kernel and generate syz programs:
```
python fuzzer.py analyze --kernel-name linux --kernel-src /path/to/linux --focus-dirs drivers/net,fs,mm --out-dir fuzz_out
```
This writes `fuzz_out/linux/analysis_summary.json` plus `programs/*.syzprog` and seed corpora.

- Stage generated programs and seeds into a syzkaller workdir:
```
python fuzzer.py stage-corpus --kernel-name linux --manager-cfg /path/to/manager.cfg --out-dir fuzz_out --fuzzer-prefix fuzzer-0
```

- One-shot (analyze + stage in one run):
```
python fuzzer.py one-shot --kernel-name linux --kernel-src /path/to/linux --manager-cfg /path/to/manager.cfg --out-dir fuzz_out --focus-dirs drivers,fs,net --fuzzer-prefix fuzzer-0
```
If `fuzz_out/<kernel>/programs` already contains prebuilt syz programs (as in this repo), `one-shot` will skip LLM preanalysis and just stage them.

- Train/mutate corpora with VAE feedback from syzkaller stats:
```
python vae_fuzzing.py --kernel linux --out-dir fuzz_out --syzkaller-workdir /path/to/workdir --stats-file /path/to/workdir/fuzzer.stats --epochs 100 --fuzz-seconds 60
```
The loop mutates `workdir/corpus/<prefix>` based on coverage/crash metrics parsed from syzkaller stats/log files.


