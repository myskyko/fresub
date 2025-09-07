#include "cuda_kernels.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cstdint>

// Need to include Window definition - must match the real struct layout
// This is a bit hacky but avoids pulling in the full AIG dependencies
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

// CUDA kernel version of solve_resub_overlap using flattened array with variable divisor count
__device__ int solve_resub_overlap_cuda(int i, int j, int k, int l, uint64_t *flat_problem, int nWords, int problem_offset, int n_divs) 
{
    // Return 1 if feasible, 0 otherwise (avoid bit shifts beyond 31)
    int res = 1;
    uint64_t qs[32] = {0};
    
    for (int h = 0; h < nWords; h++) {
        // Access flattened array: flat_problem[problem_offset + row_id * nWords + word_id]
        uint64_t t_i = flat_problem[problem_offset + i * nWords + h];
        uint64_t t_j = flat_problem[problem_offset + j * nWords + h];
        uint64_t t_k = flat_problem[problem_offset + k * nWords + h];
        uint64_t t_l = flat_problem[problem_offset + l * nWords + h];
        // Target truth table is at position n_divs (last element)
        uint64_t t_on = flat_problem[problem_offset + n_divs * nWords + h];
        uint64_t t_off = ~t_on;  // Compute off-set from on-set like CPU version

        qs[0]  |=  (t_off &  t_i &  t_j) & ( t_k &  t_l);
        qs[1]  |=  (t_on  &  t_i &  t_j) & ( t_k &  t_l);
        qs[2]  |=  (t_off & ~t_i &  t_j) & ( t_k &  t_l);
        qs[3]  |=  (t_on  & ~t_i &  t_j) & ( t_k &  t_l);
        qs[4]  |=  (t_off &  t_i & ~t_j) & ( t_k &  t_l);
        qs[5]  |=  (t_on  &  t_i & ~t_j) & ( t_k &  t_l);
        qs[6]  |=  (t_off & ~t_i & ~t_j) & ( t_k &  t_l);
        qs[7]  |=  (t_on  & ~t_i & ~t_j) & ( t_k &  t_l);
        qs[8]  |=  (t_off &  t_i &  t_j) & (~t_k &  t_l);
        qs[9]  |=  (t_on  &  t_i &  t_j) & (~t_k &  t_l);
        qs[10] |=  (t_off & ~t_i &  t_j) & (~t_k &  t_l);
        qs[11] |=  (t_on  & ~t_i &  t_j) & (~t_k &  t_l);
        qs[12] |=  (t_off &  t_i & ~t_j) & (~t_k &  t_l);
        qs[13] |=  (t_on  &  t_i & ~t_j) & (~t_k &  t_l);
        qs[14] |=  (t_off & ~t_i & ~t_j) & (~t_k &  t_l);
        qs[15] |=  (t_on  & ~t_i & ~t_j) & (~t_k &  t_l);
        qs[16] |=  (t_off &  t_i &  t_j) & ( t_k & ~t_l);
        qs[17] |=  (t_on  &  t_i &  t_j) & ( t_k & ~t_l);
        qs[18] |=  (t_off & ~t_i &  t_j) & ( t_k & ~t_l);
        qs[19] |=  (t_on  & ~t_i &  t_j) & ( t_k & ~t_l);
        qs[20] |=  (t_off &  t_i & ~t_j) & ( t_k & ~t_l);
        qs[21] |=  (t_on  &  t_i & ~t_j) & ( t_k & ~t_l);
        qs[22] |=  (t_off & ~t_i & ~t_j) & ( t_k & ~t_l);
        qs[23] |=  (t_on  & ~t_i & ~t_j) & ( t_k & ~t_l);
        qs[24] |=  (t_off &  t_i &  t_j) & (~t_k & ~t_l);
        qs[25] |=  (t_on  &  t_i &  t_j) & (~t_k & ~t_l);
        qs[26] |=  (t_off & ~t_i &  t_j) & (~t_k & ~t_l);
        qs[27] |=  (t_on  & ~t_i &  t_j) & (~t_k & ~t_l);
        qs[28] |=  (t_off &  t_i & ~t_j) & (~t_k & ~t_l);
        qs[29] |=  (t_on  &  t_i & ~t_j) & (~t_k & ~t_l);
        qs[30] |=  (t_off & ~t_i & ~t_j) & (~t_k & ~t_l);
        qs[31] |=  (t_on  & ~t_i & ~t_j) & (~t_k & ~t_l);
    }
    
    for (int h = 0; h < 16; h++) {
        int fail = ((qs[2*h] != 0) && (qs[2*h+1] != 0));
        if (fail) { res = 0; }
    }
    return res;
}

// CUDA kernel: Thread I handles i=I%THREADS_PER_PROBLEM for problem I/THREADS_PER_PROBLEM
// Now supports variable number of divisors and inputs per problem with no wasted space
// problem_offsets has M+1 elements where problem_offsets[M] = total_elements
__global__ void solve_resub_problems_kernel(uint64_t *flat_problems, int *solutions, 
                                           int *problem_offsets, int *num_inputs, int M) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int problem_id = global_tid / THREADS_PER_PROBLEM;
    int thread_in_problem = global_tid % THREADS_PER_PROBLEM;
    
    if (problem_id >= M) return;
    
    int N = num_inputs[problem_id];  // Get number of inputs for this problem
    int size = 1 << N;
    int nWords = size / word_width + (size % word_width != 0 ? 1 : 0);
    int problem_offset = problem_offsets[problem_id];  // Get offset where this problem's data starts
    
    // Compute divisor count from offsets (problem_offsets[M] contains total_elements)
    int n_divs = (problem_offsets[problem_id + 1] - problem_offset) / nWords - 1;  // -1 for target
    
    // Local storage for first feasible tuple this thread finds (keep first only)
    int li = -1, lj = -1, lk = -1, ll = -1;

    // Each thread handles multiple values of i using stride pattern
    // Loop only up to actual number of divisors for this problem
    for (int i = thread_in_problem; i < n_divs; i += THREADS_PER_PROBLEM) {
        for (int j = i + 1; j < n_divs; j++) {
            for (int k = j + 1; k < n_divs; k++) {
                for (int l = k + 1; l < n_divs; l++) {
                    int ok = solve_resub_overlap_cuda(i, j, k, l, flat_problems, nWords, problem_offset, n_divs);
                    if (ok && li == -1) {
                        li = i; lj = j; lk = k; ll = l;
                    }
                }
            }
        }
    }
    
    // Reduction within threads working on same problem (shared memory per block)
    __shared__ int shared_i[BLOCK_SIZE];
    __shared__ int shared_j[BLOCK_SIZE];
    __shared__ int shared_k[BLOCK_SIZE];
    __shared__ int shared_l[BLOCK_SIZE];
    int shared_idx = threadIdx.x;
    shared_i[shared_idx] = li;
    shared_j[shared_idx] = lj;
    shared_k[shared_idx] = lk;
    shared_l[shared_idx] = ll;
    __syncthreads();
    
    // First thread of each problem group performs reduction
    if (thread_in_problem == 0) {
        int found_any = 0;
        int oi = -1, oj = -1, okk = -1, ol = -1;
        for (int t = 0; t < THREADS_PER_PROBLEM; t++) {
            int idx = (threadIdx.x / THREADS_PER_PROBLEM) * THREADS_PER_PROBLEM + t;
            if (!found_any && shared_i[idx] != -1) {
                found_any = 1;
                oi = shared_i[idx];
                oj = shared_j[idx];
                okk = shared_k[idx];
                ol = shared_l[idx];
            }
        }
        // Write 4 integers per problem (or -1s if none)
        solutions[4 * problem_id + 0] = oi;
        solutions[4 * problem_id + 1] = oj;
        solutions[4 * problem_id + 2] = okk;
        solutions[4 * problem_id + 3] = ol;
    }
}


// Host function to launch CUDA kernel - takes pre-flattened array with offsets
// problem_offsets has M+1 elements where problem_offsets[M] = total_elements
void solve_resub_problems_cuda(uint64_t *flat_problems, int *solutions, 
                              int *problem_offsets, int *num_inputs, int M, int total_elements) {
    // Allocate device memory
    uint64_t *d_flat_problems;
    int *d_solutions;
    int *d_problem_offsets;
    int *d_num_inputs;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_flat_problems, total_elements * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_solutions, 4 * M * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_problem_offsets, (M + 1) * sizeof(int)));  // M+1 elements
    CHECK_CUDA_ERROR(cudaMalloc(&d_num_inputs, M * sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_flat_problems, flat_problems, total_elements * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_problem_offsets, problem_offsets, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_num_inputs, num_inputs, M * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel - adjust for multiple threads per problem
    int blockSize = BLOCK_SIZE;
    int totalThreads = M * THREADS_PER_PROBLEM;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    
    solve_resub_problems_kernel<<<numBlocks, blockSize>>>(d_flat_problems, d_solutions, 
                                                          d_problem_offsets, d_num_inputs, M);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back (4 ints per problem)
    CHECK_CUDA_ERROR(cudaMemcpy(solutions, d_solutions, 4 * M * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_flat_problems));
    CHECK_CUDA_ERROR(cudaFree(d_solutions));
    CHECK_CUDA_ERROR(cudaFree(d_problem_offsets));
    CHECK_CUDA_ERROR(cudaFree(d_num_inputs));
}

// Convert solution mask to vector of divisor indices
std::vector<int> mask_to_indices(uint32_t mask) {
    std::vector<int> indices;
    for (int i = 0; i < 32; i++) {
        if (mask & (1U << i)) {
            indices.push_back(i);
        }
    }
    return indices;
}

} // namespace cuda
} // namespace fresub

namespace fresub {

// CUDA-compatible feasibility check function with vector iterator interface (original version)
void feasibility_check_cuda(std::vector<Window>::iterator begin, std::vector<Window>::iterator end) {
    int M = std::distance(begin, end);  // Number of problems (windows)
    if (M == 0) return;
    
    // Calculate total size needed and build offset array
    std::vector<int> problem_offsets(M + 1);  // M+1 elements
    std::vector<int> num_inputs(M);
    int total_elements = 0;
    
    int idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
        int N = it->inputs.size();
        num_inputs[idx] = N;
        int size = 1 << N;
        int nWords = (size + 63) / 64;  // More standard rounding up
        int n_truth_tables = it->truth_tables.size();  // divisors + target
        
        problem_offsets[idx] = total_elements;
        total_elements += n_truth_tables * nWords;
    }
    problem_offsets[M] = total_elements;  // Last element points to end
    
    // Allocate flattened array with exact size needed
    std::vector<uint64_t> flat_problems(total_elements, 0);
    
    // Flatten all windows into the array
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
    
    // Allocate solutions array: 4 integers per problem (i,j,k,l) or -1s if none
    std::vector<int> solutions(4 * M, -1);
    
    // Call CUDA kernel (finds one feasible combination per problem if any)
    cuda::solve_resub_problems_cuda(flat_problems.data(), solutions.data(), 
                                    problem_offsets.data(), num_inputs.data(), 
                                    M, total_elements);
    
    // Convert results back to feasible sets for each window
    idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
        int i0 = solutions[4 * idx + 0];
        int j0 = solutions[4 * idx + 1];
        int k0 = solutions[4 * idx + 2];
        int l0 = solutions[4 * idx + 3];
        if (i0 >= 0 && j0 >= 0 && k0 >= 0 && l0 >= 0) {
            FeasibleSet fs;
            fs.divisor_indices = {i0, j0, k0, l0};
            it->feasible_sets.push_back(std::move(fs));
        }
    }
}

} // namespace fresub
