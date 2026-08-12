#include "cuda_kernels.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cassert>

// Need to include Window definition - must match the real struct layout
namespace fresub {
struct aigman; // forward declaration for pointer use
struct FeasibleSet {
    std::vector<int> divisor_indices;
    std::vector<aigman*> synths;
};
struct Window {
    int target_node;
    std::vector<int> inputs;     // Window inputs (cut leaves)
    std::vector<int> nodes;      // All nodes in window
    std::vector<int> divisors;   // Window nodes - MFFC(target)
    int cut_id;                  // ID of the cut that generated this window
    int mffc_size;
    std::vector<std::vector<uint64_t>> truth_tables;
    std::vector<FeasibleSet> feasible_sets;
};
}

#define word_width 64
#define THREADS_PER_PROBLEM 32
#define BLOCK_SIZE 256

// Check for CUDA error macro
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
#endif

namespace fresub {
namespace cuda {

// External declaration - function defined in resub_kernels.cu
extern __device__ int solve_resub_overlap_cuda(int i, int j, int k, int l, uint64_t *flat_problem, int nWords, int problem_offset, int n_divs);

// CUDA kernel to mark all feasible 4-combinations as boolean array
// Each thread handles multiple i values for one problem using the same loop structure
__global__ void solve_resub_problems_kernel_all(uint64_t *flat_problems, char *feasibility_results,
                                               int *problem_offsets, unsigned long long *combination_offsets,
                                               int *num_inputs, int M) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int problem_id = global_tid / THREADS_PER_PROBLEM;
    int thread_in_problem = global_tid % THREADS_PER_PROBLEM;
    
    if (problem_id >= M) return;
    
    int N = num_inputs[problem_id];
    int size = 1 << N;
    int nWords = size / word_width + (size % word_width != 0 ? 1 : 0);
    int problem_offset = problem_offsets[problem_id];
    int n_divs = (problem_offsets[problem_id + 1] - problem_offset) / nWords - 1;
    
    unsigned long long combination_base = combination_offsets[problem_id];
    
    // Each thread handles multiple values of i using stride pattern (same as original)
    for (int i = thread_in_problem; i < n_divs; i += THREADS_PER_PROBLEM) {
        for (int j = i + 1; j < n_divs; j++) {
            for (int k = j + 1; k < n_divs; k++) {
                for (int l = k + 1; l < n_divs; l++) {
                    // Calculate index in feasibility array using 4D indexing
                    unsigned long long nd = static_cast<unsigned long long>(n_divs);
                    unsigned long long combination_idx =
                        static_cast<unsigned long long>(l) +
                        static_cast<unsigned long long>(k) * nd +
                        static_cast<unsigned long long>(j) * nd * nd +
                        static_cast<unsigned long long>(i) * nd * nd * nd;
                    unsigned long long global_idx = combination_base + combination_idx;
                    
                    uint32_t mask = solve_resub_overlap_cuda(i, j, k, l, flat_problems, nWords, problem_offset, n_divs);
                    feasibility_results[global_idx] = (mask != 0) ? 1 : 0;
                }
            }
        }
    }
}

// Host function to launch new CUDA kernel that finds all feasible sets
void solve_resub_problems_cuda_all(uint64_t *flat_problems, char *feasibility_results,
                                   int *problem_offsets, unsigned long long *combination_offsets,
                                   int *num_inputs, int M, int total_elements, unsigned long long total_combinations) {
    // Allocate device memory
    uint64_t *d_flat_problems;
    char *d_feasibility_results;
    int *d_problem_offsets;
    unsigned long long *d_combination_offsets;
    int *d_num_inputs;

    CHECK_CUDA_ERROR(cudaMalloc(&d_flat_problems, total_elements * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_feasibility_results, total_combinations * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_problem_offsets, (M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_combination_offsets, (M + 1) * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_num_inputs, M * sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_flat_problems, flat_problems, total_elements * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_problem_offsets, problem_offsets, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_combination_offsets, combination_offsets, (M + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_num_inputs, num_inputs, M * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = BLOCK_SIZE;
    int totalThreads = M * THREADS_PER_PROBLEM;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    solve_resub_problems_kernel_all<<<numBlocks, blockSize>>>(d_flat_problems, d_feasibility_results,
                                                              d_problem_offsets, d_combination_offsets, 
                                                              d_num_inputs, M);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(feasibility_results, d_feasibility_results, total_combinations * sizeof(char), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_flat_problems));
    CHECK_CUDA_ERROR(cudaFree(d_feasibility_results));
    CHECK_CUDA_ERROR(cudaFree(d_problem_offsets));
    CHECK_CUDA_ERROR(cudaFree(d_combination_offsets));
    CHECK_CUDA_ERROR(cudaFree(d_num_inputs));
}

} // namespace cuda
} // namespace fresub

namespace fresub {

// CUDA-compatible feasibility check function that finds ALL feasible sets
void feasibility_check_cuda_all(std::vector<Window>::iterator begin, std::vector<Window>::iterator end) {
    int M = std::distance(begin, end);  // Number of problems (windows)
    if (M == 0) return;
    
    // Calculate total size needed for truth tables and build offset arrays
    std::vector<int> problem_offsets(M + 1);  // M+1 elements
    std::vector<unsigned long long> combination_offsets(M + 1);  // M+1 elements
    std::vector<int> num_inputs(M);
    int total_elements = 0;
    unsigned long long total_combinations = 0;
    
    int idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
        int N = it->inputs.size();
        num_inputs[idx] = N;
        int size = 1 << N;
        int nWords = (size + 63) / 64;  // More standard rounding up
        int n_truth_tables = it->truth_tables.size();  // divisors + target
        int n_divs = n_truth_tables - 1;  // -1 for target
        
        problem_offsets[idx] = total_elements;
        total_elements += n_truth_tables * nWords;
        
        combination_offsets[idx] = total_combinations;
        unsigned long long nd = static_cast<unsigned long long>(n_divs);
        total_combinations += nd * nd * nd * nd;  // D^4 combinations
    }
    problem_offsets[M] = total_elements;  // Last element points to end
    combination_offsets[M] = total_combinations;  // Last element points to end
    
    // Allocate flattened arrays
    std::vector<uint64_t> flat_problems(total_elements, 0);
    std::vector<char> feasibility_results(total_combinations, 0);  // Use char instead of bool
    
    // Flatten all windows into the truth table array
    idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
        int N = it->inputs.size();
        int size = 1 << N;
        int nWords = size / 64 + (size % 64 != 0 ? 1 : 0);
        int problem_offset = problem_offsets[idx];
        
        // Copy all truth tables (divisors + target)
        for (size_t t = 0; t < it->truth_tables.size(); t++) {
            for (int w = 0; w < nWords; w++) {
                if (w < (int)it->truth_tables[t].size()) {
                    flat_problems[problem_offset + t * nWords + w] = it->truth_tables[t][w];
                }
            }
        }
    }
    // Call CUDA kernel to find all feasible sets
    cuda::solve_resub_problems_cuda_all(flat_problems.data(), feasibility_results.data(),
                                        problem_offsets.data(), combination_offsets.data(),
                                        num_inputs.data(), M, total_elements, total_combinations);
    
    // Convert results back to feasible sets for each window
    idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
        int n_divs = it->truth_tables.size() - 1;  // -1 for target
        unsigned long long combination_base = combination_offsets[idx];
        
        assert(it->feasible_sets.empty());
        
        // Check all valid combinations (i < j < k < l)
        for (int i = 0; i < n_divs; i++) {
            for (int j = i + 1; j < n_divs; j++) {
                for (int k = j + 1; k < n_divs; k++) {
                    for (int l = k + 1; l < n_divs; l++) {
                        unsigned long long nd = static_cast<unsigned long long>(n_divs);
                        unsigned long long combination_idx =
                            static_cast<unsigned long long>(l) +
                            static_cast<unsigned long long>(k) * nd +
                            static_cast<unsigned long long>(j) * nd * nd +
                            static_cast<unsigned long long>(i) * nd * nd * nd;
                        unsigned long long global_idx = combination_base + combination_idx;
                        
                        if (feasibility_results[global_idx]) {
                            FeasibleSet fs;
                            fs.divisor_indices = {i, j, k, l};
                            it->feasible_sets.push_back(std::move(fs));
                        }
                    }
                }
            }
        }
    }
}

} // namespace fresub
