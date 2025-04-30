#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for SuppressTokens
__global__ void suppress_tokens_kernel(
    float* logits, int batch_size, int vocab_size,
    const int* suppress_tokens, int num_suppress
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    if (batch < batch_size && tid < num_suppress) {
        int token = suppress_tokens[tid];
        if (token >= 0 && token < vocab_size) {
            logits[batch * vocab_size + token] = -INFINITY;
        }
    }
}

// CUDA kernel for timestamp filtering (ApplyTimestampRules)
__global__ void apply_timestamp_rules_kernel(
    float* logits, int batch_size, int vocab_size,
    const int* tokens, int seq_len, int sample_begin,
    int timestamp_begin, int eot_token, int no_timestamps_token,
    int max_initial_timestamp_index, bool is_first_step
) {
    int batch = blockIdx.x;
    float* logit_row = logits + batch * vocab_size;
    // 1. Mask <|notimestamps|>
    if (no_timestamps_token >= 0 && no_timestamps_token < vocab_size) {
        logit_row[no_timestamps_token] = -INFINITY;
    }
    // 2. Mask at first step
    if (is_first_step) {
        for (int i = 0; i < timestamp_begin; ++i) logit_row[i] = -INFINITY;
        if (max_initial_timestamp_index >= 0) {
            int last_allowed = timestamp_begin + max_initial_timestamp_index;
            for (int i = last_allowed + 1; i < vocab_size; ++i) logit_row[i] = -INFINITY;
        }
        return;
    }
    // TODO: Implement the rest of the timestamp pair/decreasing logic
}

// Main entry point for logit filtering (currently only SuppressTokens)
void cuda_logit_filter(
    torch::Tensor logits, // [batch_size, vocab_size], float32, CUDA
    torch::Tensor suppress_tokens, // [num_suppress], int32, CUDA
    torch::Tensor tokens, // [batch_size, seq_len], int32, CUDA (optional, can be empty)
    int sample_begin,
    int timestamp_begin,
    int eot_token,
    int no_timestamps_token,
    int max_initial_timestamp_index,
    bool is_first_step
) {
    int batch_size = logits.size(0);
    int vocab_size = logits.size(1);
    int num_suppress = suppress_tokens.size(0);
    float* logits_ptr = logits.data_ptr<float>();
    const int* suppress_ptr = suppress_tokens.data_ptr<int>();
    int blockSize = num_suppress;
    suppress_tokens_kernel<<<batch_size, blockSize>>>(logits_ptr, batch_size, vocab_size, suppress_ptr, num_suppress);
    // Timestamp filtering (if tokens is not empty)
    if (tokens.defined() && tokens.numel() > 0) {
        int seq_len = tokens.size(1);
        const int* tokens_ptr = tokens.data_ptr<int>();
        apply_timestamp_rules_kernel<<<batch_size, 1>>>(
            logits_ptr, batch_size, vocab_size,
            tokens_ptr, seq_len, sample_begin,
            timestamp_begin, eot_token, no_timestamps_token,
            max_initial_timestamp_index, is_first_step
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_logit_filter", &cuda_logit_filter, "CUDA Logit Filter (SuppressTokens + TimestampRules)");
} 