#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Helper struct for candidate
struct Candidate {
    float score;
    int beam_idx;
    int token;
    __host__ __device__ bool operator<(const Candidate& other) const {
        return score > other.score; // descending
    }
};

__global__ void beam_search_update_kernel(
    const float* logits,      // [num_beams, vocab_size]
    const int* tokens,        // [num_beams, seq_len]
    const float* sum_logprobs,// [num_beams]
    int* out_tokens,          // [num_beams, seq_len+1]
    float* out_sum_logprobs,  // [num_beams]
    int* out_source_indices,  // [num_beams]
    int* out_finished,        // [num_beams]
    int num_audio,
    int beam_size,
    int vocab_size,
    int seq_len,
    int eot_token
) {
    int audio_idx = blockIdx.x;
    if (audio_idx >= num_audio) return;

    int base = audio_idx * beam_size;
    int candidate_count = beam_size * vocab_size;

    // Allocate candidates in global memory
    Candidate* candidates = new Candidate[candidate_count];

    int cand_idx = 0;
    for (int j = 0; j < beam_size; ++j) {
        int beam_idx = base + j;
        for (int k = 0; k < vocab_size; ++k) {
            float logprob = logits[beam_idx * vocab_size + k];
            float score = sum_logprobs[beam_idx] + logprob;
            candidates[cand_idx++] = {score, beam_idx, k};
        }
    }

    // Selection sort for top beam_size candidates
    for (int i = 0; i < beam_size; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < candidate_count; ++j) {
            if (candidates[j].score > candidates[max_idx].score) {
                max_idx = j;
            }
        }
        // Swap
        if (max_idx != i) {
            Candidate temp = candidates[i];
            candidates[i] = candidates[max_idx];
            candidates[max_idx] = temp;
        }
    }

    // Output top beam_size candidates
    for (int i = 0; i < beam_size; ++i) {
        int out_idx = base + i;
        int src_beam = candidates[i].beam_idx;
        int token = candidates[i].token;
        float score = candidates[i].score;
        // Copy previous tokens
        for (int t = 0; t < seq_len; ++t) {
            out_tokens[out_idx * (seq_len+1) + t] = tokens[src_beam * seq_len + t];
        }
        out_tokens[out_idx * (seq_len+1) + seq_len] = token;
        out_sum_logprobs[out_idx] = score;
        out_source_indices[out_idx] = src_beam;
        out_finished[out_idx] = (token == eot_token) ? 1 : 0;
    }

    delete[] candidates;
}

// C++/PyTorch binding
void beam_search_update(
    torch::Tensor logits,
    torch::Tensor tokens,
    torch::Tensor sum_logprobs,
    torch::Tensor out_tokens,
    torch::Tensor out_sum_logprobs,
    torch::Tensor out_source_indices,
    torch::Tensor out_finished,
    int num_audio,
    int beam_size,
    int vocab_size,
    int seq_len,
    int eot_token
) {
    int blocks = num_audio;
    int threads = 1;
    beam_search_update_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        tokens.data_ptr<int>(),
        sum_logprobs.data_ptr<float>(),
        out_tokens.data_ptr<int>(),
        out_sum_logprobs.data_ptr<float>(),
        out_source_indices.data_ptr<int>(),
        out_finished.data_ptr<int>(),
        num_audio,
        beam_size,
        vocab_size,
        seq_len,
        eot_token
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beam_search_update", &beam_search_update, "Beam Search Update (CUDA)");
} 