import torch
from torch.utils.cpp_extension import load
import os

# Load the CUDA extension (builds on first import)
ext_path = os.path.join(os.path.dirname(__file__), '..', 'CUDA', 'topk_beam_expansion_kernel.cu')
topk_beam_expansion_cuda = load(
    name="topk_beam_expansion_cuda_ext",
    sources=[ext_path],
    verbose=True,
    extra_cuda_cflags=["--expt-extended-lambda"]
)

def cuda_topk_beam_expansion(tokens, logits, sum_logprobs, beam_size, eot_token):
    # tokens: [num_beams, seq_len]
    # logits: [num_beams, vocab_size]
    # sum_logprobs: [num_beams]
    num_beams, vocab_size = logits.shape
    seq_len = tokens.shape[1]
    num_audio = num_beams // beam_size

    out_tokens = torch.empty((num_beams, seq_len+1), dtype=torch.int32, device=tokens.device)
    out_sum_logprobs = torch.empty((num_beams,), dtype=torch.float32, device=tokens.device)
    out_source_indices = torch.empty((num_beams,), dtype=torch.int32, device=tokens.device)
    out_finished = torch.empty((num_beams,), dtype=torch.int32, device=tokens.device)

    topk_beam_expansion_cuda.topk_beam_expansion(
        logits, tokens, sum_logprobs, out_tokens, out_sum_logprobs, out_source_indices, out_finished,
        num_audio, beam_size, vocab_size, seq_len, beam_size, eot_token
    )

    return out_tokens, out_sum_logprobs, out_source_indices, out_finished 