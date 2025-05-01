# CUDA Beam Search

This module provides CUDA-accelerated beam search operations for use with Whisper and PyTorch models. It includes custom CUDA kernels and Python bindings for efficient sequence decoding, particularly for speech recognition tasks.

## Structure
- `CUDA/`: Contains CUDA kernel implementations.
- `cuda_beam_search/`: Python bindings and utilities.

## Requirements
See `requirements.txt` for dependencies.

## Usage
Import and use the CUDA beam search functions in your Whisper-based pipelines for faster decoding.

---

For development or troubleshooting, see the source files and ensure your environment meets the requirements. 