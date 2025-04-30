import torch
from torch.utils.cpp_extension import load
import os

ext_path = os.path.join(os.path.dirname(__file__), '..', 'CUDA', 'logit_filter_kernel.cu')
logit_filter_cuda = load(
    name="logit_filter_cuda_ext",
    sources=[ext_path],
    verbose=True,
    extra_cuda_cflags=["--expt-extended-lambda"]
)

def cuda_logit_filter(
    logits,
    suppress_tokens,
    tokens=None,
    sample_begin=0,
    timestamp_begin=0,
    eot_token=0,
    no_timestamps_token=-1,
    max_initial_timestamp_index=-1,
    is_first_step=False
):
    """
    GPU logit filter for suppress tokens and timestamp rules.
    Args:
        logits: [batch_size, vocab_size] (float32, CUDA)
        suppress_tokens: [num_suppress] (int32, CUDA)
        tokens: [batch_size, seq_len] (int32, CUDA) or None
        sample_begin: int
        timestamp_begin: int
        eot_token: int
        no_timestamps_token: int
        max_initial_timestamp_index: int
        is_first_step: bool
    """
    if tokens is None:
        tokens = torch.empty(0, 0, dtype=torch.int32, device=logits.device)
    logit_filter_cuda.cuda_logit_filter(
        logits, suppress_tokens, tokens, sample_begin, timestamp_begin, eot_token,
        no_timestamps_token, max_initial_timestamp_index, is_first_step
    ) 