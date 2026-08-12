#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <aig.hpp>

#include "aig_utils.hpp"
#include "feasibility.hpp"
#include "insertion.hpp"
#include "simulation.hpp"
#include "window.hpp"

using namespace fresub;
using namespace std::chrono;

struct Config {
  std::string input_file;
  std::string output_file;
  int max_inputs = 4;
  int max_nodes = 64;
  int max_divisors = 0;
  unsigned long long cuda_max_result_bytes = 0;
  int k_windows = 100;
  unsigned int rng_seed = 42;
  bool verbose = false;
  bool show_stats = false;
  bool use_cuda = false;
  bool use_cuda_all = false;
  bool feas_all = false;
};

static unsigned long long cuda_all_result_bytes(const Window& window) {
  const unsigned long long n_divs = window.truth_tables.empty() ? 0ull : window.truth_tables.size() - 1ull;
  return n_divs * n_divs * n_divs * n_divs;
}

int main(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      config.verbose = true;
    } else if (strcmp(argv[i], "-s") == 0) {
      config.show_stats = true;
    } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      config.max_inputs = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      config.max_nodes = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--max-divisors") == 0 && i + 1 < argc) {
      config.max_divisors = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--cuda-max-result-mb") == 0 && i + 1 < argc) {
      config.cuda_max_result_bytes = std::strtoull(argv[++i], nullptr, 10) * 1024ull * 1024ull;
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      config.k_windows = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--cuda") == 0) {
      config.use_cuda = true;
    } else if (strcmp(argv[i], "--cuda-all") == 0) {
      config.use_cuda_all = true;
    } else if (strcmp(argv[i], "--feas-all") == 0) {
      config.feas_all = true;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      char* endp = nullptr;
      unsigned long v = std::strtoul(argv[++i], &endp, 10);
      config.rng_seed = static_cast<unsigned int>(v);
    } else if (argv[i][0] != '-') {
      if (config.input_file.empty()) {
        config.input_file = argv[i];
      } else if (config.output_file.empty()) {
        config.output_file = argv[i];
      }
    }
  }
  if (config.input_file.empty()) {
    std::cerr << "Usage: " << argv[0] << " [options] <input.blif> [output.blif]\n";
    std::cerr << "Options:\n";
    std::cerr << "  -i <N>        Max window boundary inputs (default: 4)\n";
    std::cerr << "  -n <N>        Max graph nodes per window (default: 64)\n";
    std::cerr << "  --max-divisors <N>  Cap divisors per window before simulation/feasibility (default: unlimited)\n";
    std::cerr << "  --cuda-max-result-mb <N>  Max CUDA-all result bytes per batch in MB (default: unlimited)\n";
    std::cerr << "  -k <K>        Process K random windows (default: 100)\n";
    std::cerr << "  --seed <N>    RNG seed for -k sampling (default: 42)\n";
    std::cerr << "  -v            Verbose output\n";
    std::cerr << "  -s            Show statistics\n";
    std::cerr << "  --cuda        Use CUDA for 4-LUT feasibility checking (first solution)\n";
    std::cerr << "  --cuda-all    Use CUDA for 4-LUT feasibility checking (all solutions)\n";
    std::cerr << "  --feas-all    CPU feasibility: ALL mode (default is MIN-SIZE)\n";
    return 1;
  }
  aigman aig;
  if (!read_blif_as_cover_aig(config.input_file, aig, config.verbose)) {
    std::cerr << "Failed to read BLIF: " << config.input_file << "\n";
    return 1;
  }

  const size_t initial_luts = aig.vCoverRoots.size();
  if (config.show_stats || config.verbose) {
    std::cout << "Initial AIG: " << aig.nPis << " PIs, " << aig.nPos << " POs, " << initial_luts << " LUTs\n";
  }
  if (config.verbose) {
    if (config.use_cuda_all) {
      std::cout << "Using CUDA feasibility checking (all 4-divisor combinations)\n";
    } else if (config.use_cuda) {
      std::cout << "Using CUDA feasibility checking (first 4-divisor combination)\n";
    } else if (config.feas_all) {
      std::cout << "Using CPU feasibility (ALL mode)\n";
    } else {
      std::cout << "Using CPU feasibility (MIN-SIZE mode)\n";
    }
  }

  auto start_time = high_resolution_clock::now();

  BfsWindowParams ps;
  ps.max_inputs = config.max_inputs;
  ps.max_nodes = config.max_nodes;

  std::vector<Window> windows;
  window_extract_lut_bfs(aig, ps, config.verbose, windows);
  const size_t total_extracted = windows.size();

  if (config.k_windows > 0 && windows.size() > static_cast<size_t>(config.k_windows)) {
    std::vector<size_t> idx(windows.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(config.rng_seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<Window> selected;
    selected.reserve(static_cast<size_t>(config.k_windows));
    for (int i = 0; i < config.k_windows; ++i) {
      selected.push_back(std::move(windows[idx[static_cast<size_t>(i)]]));
    }
    windows.swap(selected);
  }

  for (auto& window : windows) {
    if (config.max_divisors > 0 && window.divisors.size() > static_cast<size_t>(config.max_divisors)) {
      window.divisors.resize(static_cast<size_t>(config.max_divisors));
    }
  }

  for (auto& window : windows) {
    window.truth_tables = compute_truth_tables_for_window(aig, window, config.verbose);
  }

  if (config.use_cuda_all) {
    if (config.cuda_max_result_bytes == 0) {
      feasibility_check_cuda_all(windows.begin(), windows.end());
    } else {
      auto batch_begin = windows.begin();
      while (batch_begin != windows.end()) {
        auto batch_end = batch_begin;
        unsigned long long batch_bytes = 0;
        while (batch_end != windows.end()) {
          const unsigned long long window_bytes = cuda_all_result_bytes(*batch_end);
          if (batch_end != batch_begin && batch_bytes + window_bytes > config.cuda_max_result_bytes) {
            break;
          }
          batch_bytes += window_bytes;
          ++batch_end;
          if (window_bytes > config.cuda_max_result_bytes) {
            break;
          }
        }
        feasibility_check_cuda_all(batch_begin, batch_end);
        batch_begin = batch_end;
      }
    }
  } else if (config.use_cuda) {
    feasibility_check_cuda(windows.begin(), windows.end());
  } else if (config.feas_all) {
    feasibility_check_cpu_all(windows.begin(), windows.end());
  } else {
    feasibility_check_cpu_min(windows.begin(), windows.end());
  }

  const int successful_resubs = inserter_process_lut_windows(aig, windows, config.verbose);

  auto end_time = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end_time - start_time);
  const size_t final_luts = aig.vCoverRoots.size();

  if (config.show_stats || config.verbose) {
    std::cout << "\nLUT resubstitution complete:\n";
    std::cout << "  Windows extracted: " << total_extracted << "\n";
    std::cout << "  Windows processed: " << windows.size() << "\n";
    std::cout << "  Successful resubstitutions: " << successful_resubs << "\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Initial LUTs: " << initial_luts << "\n";
    std::cout << "  Final LUTs: " << final_luts << "\n";
  }

  if (!config.output_file.empty()) {
    aig.write_blif(config.output_file);
  }

  return 0;
}
