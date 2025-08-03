# FuzzyGan: A Variational Autoencoder-Based Direct Fuzzing Framework
## Overview
VAE-Fuzz is a modular, extensible fuzzing framework designed to enhance the fuzzing process for C/C++ libraries using a Variational Autoencoder (VAE) for intelligent input generation. It integrates with OSS-Fuzz, leveraging its build.sh scripts to compile libraries and analyze their functions for fuzzing suitability. The framework recursively scans header files (.h and .hpp) in a specified directory, generates LibFuzzer harnesses using Gemini's generative AI, and employs VAE to mutate seed corpora based on coverage feedback.

## Project Structure
The project is organized into the following files:
- fuzzer.py: Main CLI entry point for library analysis, fuzzing, and 	  listing available/analyzed libraries.
- preanalyze.py: Handles recursive header file analysis and AI-driven harness/seed generation using Gemini.
- prompt.txt: Template for prompting Gemini to generate LibFuzzer harnesses and seeds.
- vae_model.py: Defines the VAE model and loss function for input generation.
- corpus_manager.py: Manages corpus loading, cleaning, and saving, including backup and zip archival.
- fuzzer_runner.py: Executes fuzzers and computes coverage-based loss for VAE training.
- vae_fuzzing.py: Implements VAE-based fuzzing with coverage feedback for a specific function.

## Requirements

- Python: 3.8+
- Dependencies: torch, google-generativeai, numpy
- Environment: Set GEMINI_API_KEY for Gemini API access.
- OSS-Fuzz: A local clone of library or code you want to fuzz
- Compiler: clang with AddressSanitizer and fuzzing support.

Install dependencies:
```python pip install torch google-generativeai numpy ```

Set API key:
```export GEMINI_API_KEY="your-api-key" ```

## Usage
### Commands

List Available Libraries:Lists all libraries in oss-fuzz/projects/ with a build.sh file.
```python fuzzer.py list-libraries --oss-fuzz-dir ./oss-fuzz ```


List Analyzed Libraries:Lists libraries with an analysis_summary.json in the output directory.
```python fuzzer.py list-analyzed --out-dir fuzz_out --oss-fuzz-dir ./oss-fuzz ```


Analyze a Library:Builds the library using build.sh and analyzes all .h and .hpp files in the specified header directory.
```python fuzzer.py analyze --library zlib --header-dir ./oss-fuzz/projects/zlib --exclude-prefixes zlib,get_crc --skip-functions zlibVersion,get_crc_table --out-dir fuzz_out --oss-fuzz-dir ./oss-fuzz ```


Fuzz a Library:Compiles and runs fuzzers for all functions deemed worth fuzzing.
```python fuzzer.py fuzz --library zlib --out-dir fuzz_out --oss-fuzz-dir ./oss-fuzz ```


VAE-Based Fuzzing:Performs VAE-based fuzzing for a specific function, using coverage feedback to guide input generation.
```python vae_fuzzing.py --function uncompress --library zlib --out-dir fuzz_out --oss-fuzz-dir ./oss-fuzz ```


## Future Improvements

Support for additional fuzzing engines (e.g., AFL++).
Enhanced VAE/GAN model with dynamic input sizes.
Integration with other AI models for harness generation.
Parallel processing for faster analysis of large libraries.
          
            
          
        
  
        
    

