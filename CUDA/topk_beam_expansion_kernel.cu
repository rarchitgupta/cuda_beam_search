#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

struct Candidate {
    float score;
    int beam_idx;
    int token;
};

__device__ void insert_topk(Candidate* topk, int& k, const Candidate& cand, int K) {
    // Insert candidate into top-k array (descending order)
    int pos = k;
    for (int i = 0; i < k; ++i) {
        if (cand.score > topk[i].score) {
            pos = i;
            break;
        }
    }
    if (k < K) ++k;
    for (int i = k - 1; i > pos; --i) {
        topk[i] = topk[i - 1];
    }
    if (pos < K) topk[pos] = cand;
}

__global__ void topk_beam_expansion_kernel(
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
    int topk,                 // number of candidates to select (beam_size)
    int eot_token
) {
    int audio_idx = blockIdx.x;
    if (audio_idx >= num_audio) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int base = audio_idx * beam_size;
    int candidate_count = beam_size * vocab_size;

    // If candidate_count is too large for shared memory, fall back to serial version
    if (candidate_count > 8192) {
        if (tid == 0) {
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
            // Selection sort for top-k candidates
            for (int i = 0; i < topk; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < candidate_count; ++j) {
                    if (candidates[j].score > candidates[max_idx].score) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    Candidate temp = candidates[i];
                    candidates[i] = candidates[max_idx];
                    candidates[max_idx] = temp;
                }
            }
            // Output top-k candidates
            for (int i = 0; i < topk; ++i) {
                int out_idx = base + i;
                int src_beam = candidates[i].beam_idx;
                int token = candidates[i].token;
                float score = candidates[i].score;
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
        return;
    }

    // --- Optimized: Parallel candidate scoring and top-k selection ---
    extern __shared__ Candidate shared_topk[]; // [nthreads * topk]
    Candidate local_topk[16]; // up to topk=16
    int local_k = 0;

    // Stride over all candidates
    for (int idx = tid; idx < candidate_count; idx += nthreads) {
        int j = idx / vocab_size; // beam
        int k = idx % vocab_size; // token
        int beam_idx = base + j;
        float logprob = logits[beam_idx * vocab_size + k];
        float score = sum_logprobs[beam_idx] + logprob;
        Candidate cand = {score, beam_idx, k};
        insert_topk(local_topk, local_k, cand, topk);
    }
    // Write local top-k to shared memory
    for (int i = 0; i < local_k; ++i) {
        shared_topk[tid * topk + i] = local_topk[i];
    }
    __syncthreads();

    // Only thread 0 in block does the final reduction
    if (tid == 0) {
        int total = nthreads * topk;
        Candidate final_topk[16];
        int final_k = 0;
        for (int i = 0; i < total; ++i) {
            insert_topk(final_topk, final_k, shared_topk[i], topk);
        }
        // Output top-k
        for (int i = 0; i < topk; ++i) {
            int out_idx = base + i;
            int src_beam = final_topk[i].beam_idx;
            int token = final_topk[i].token;
            float score = final_topk[i].score;
            for (int t = 0; t < seq_len; ++t) {
                out_tokens[out_idx * (seq_len+1) + t] = tokens[src_beam * seq_len + t];
            }
            out_tokens[out_idx * (seq_len+1) + seq_len] = token;
            out_sum_logprobs[out_idx] = score;
            out_source_indices[out_idx] = src_beam;
            out_finished[out_idx] = (token == eot_token) ? 1 : 0;
        }
    }
}

void topk_beam_expansion(
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
    int topk,
    int eot_token
) {
    int blocks = num_audio;
    int threads = 128;
    size_t shared_mem = threads * topk * sizeof(Candidate);
    topk_beam_expansion_kernel<<<blocks, threads, shared_mem>>>(
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
        topk,
        eot_token
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_beam_expansion", &topk_beam_expansion, "Top-K Beam Expansion (CUDA)");
} 