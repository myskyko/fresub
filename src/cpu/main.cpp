#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <cassert>
#include <iostream>
#include <random>
#include <numeric>

#include <aig.hpp>

#include "feasibility.hpp"
#include "insertion.hpp"
#include "simulation.hpp"
#include "synthesis.hpp"
#include "window.hpp"

// Forward declare CUDA functions to avoid requiring CUDA headers when not needed
namespace fresub {
  void feasibility_check_cuda(std::vector<Window>::iterator begin, std::vector<Window>::iterator end);
  void feasibility_check_cuda_all(std::vector<Window>::iterator begin, std::vector<Window>::iterator end);
}

using namespace fresub;
using namespace std::chrono;

struct Config {
    std::string input_file;
    std::string output_file;
    int max_cut_size = 4;
    int k_windows = 100;        // Number of windows to process (randomly sampled)
    unsigned int rng_seed = 42; // Seed for random sampling (deterministic by default)
    bool verbose = false;
    bool show_stats = false;
    bool use_mockturtle = true;  // Default to mockturtle synthesis
    bool use_cuda = false;       // Default to CPU feasibility check
    bool use_cuda_all = false;   // Use CUDA to find all combinations
    bool feas_all = false;       // CPU feasibility: if true ALL, else MIN-SIZE
};


int main(int argc, char** argv) {
  // Read arguments
  Config config;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      config.verbose = true;
    } else if (strcmp(argv[i], "-s") == 0) {
      config.show_stats = true;
    } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
      config.max_cut_size = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      config.k_windows = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--exopt") == 0) {
      config.use_mockturtle = false;
    } else if (strcmp(argv[i], "--mockturtle") == 0) {
      config.use_mockturtle = true;
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
    std::cerr << "Usage: " << argv[0] << " [options] <input.aig> [output.aig]\n";
    std::cerr << "Options:\n";
    std::cerr << "  -c <size>     Max cut size (default: 4)\n";
    std::cerr << "  -k <K>        Process K random windows (default: 100)\n";
    std::cerr << "  --seed <N>    RNG seed for -k sampling (default: 42)\n";
    std::cerr << "  -v            Verbose output\n";
    std::cerr << "  -s            Show statistics\n";
    std::cerr << "  --exopt       Use SAT-based synthesis (exopt)\n";
    std::cerr << "  --mockturtle  Use library-based synthesis (mockturtle, default)\n";
    std::cerr << "  --cuda        Use CUDA for feasibility checking (first solution)\n";
    std::cerr << "  --cuda-all    Use CUDA for feasibility checking (all solutions)\n";
    std::cerr << "  --feas-all    CPU feasibility: ALL mode (default is MIN-SIZE)\n";
    return 1;
  }
  
  // Load input AIG
  if (config.verbose) {
    std::cout << "Loading AIG from " << config.input_file << "...\n";
  }
  aigman aig;
  aig.read(config.input_file.c_str());
  int initial_gates = aig.nGates;
  if (config.show_stats) {
    std::cout << "Initial AIG: " << aig.nPis << " PIs, " << aig.nPos << " POs, " << initial_gates << " gates\n";
  }
  if (config.verbose) {
    std::cout << "Using " << (config.use_mockturtle ? "mockturtle library-based" : "exopt SAT-based") << " synthesis\n";
    if (config.use_cuda_all) {
      std::cout << "Using CUDA feasibility checking (all combinations)\n";
    } else if (config.use_cuda) {
      std::cout << "Using CUDA feasibility checking (first combination)\n";
    } else if (config.feas_all) {
      std::cout << "Using CPU feasibility (ALL mode)\n";
    } else {
      std::cout << "Using CPU feasibility (MIN-SIZE mode)\n";
    }
  }

  // Start measurement
  auto start_time = high_resolution_clock::now();
        
  // Extract windows
  if (config.verbose) {
    std::cout << "Extracting windows with max cut size " << config.max_cut_size << "...\n";
  }
  std::vector<Window> windows;
  window_extract_all(aig, config.max_cut_size, config.verbose, windows);
  if (config.verbose) {
    std::cout << "Extracted " << windows.size() << " windows\n";
  }

  // Batch mode: process K randomly selected windows (default K=100)
  size_t total_extracted = windows.size();
  if (config.k_windows > 0 && windows.size() > static_cast<size_t>(config.k_windows)) {
    // Sample without replacement using a shuffled index vector
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
    if (config.verbose) {
      std::cout << "Processing a random batch of " << windows.size()
                << " windows out of " << total_extracted << " total"
                << " (seed=" << config.rng_seed << ")\n";
    }
  }

  // Previously: excluded windows with <4 divisors. Now process all windows.
  
  // Compute truth tables
  for (auto& window : windows) {
    window.truth_tables = compute_truth_tables_for_window(aig, window, config.verbose);
  }

  // Feasibility check
  if (config.use_cuda_all) {
    feasibility_check_cuda_all(windows.begin(), windows.end());
  } else if (config.use_cuda) {
    feasibility_check_cuda(windows.begin(), windows.end());
  } else if (config.feas_all) {
    feasibility_check_cpu_all(windows.begin(), windows.end());
  } else {
    feasibility_check_cpu_min(windows.begin(), windows.end());
  }
  
  // Synthesize for all feasible sets; do not pre-filter before insertion
  for (auto& window : windows) {
    if (config.verbose) {
      std::cout << "Processing window with target " << window.target_node
		<< " (" << window.inputs.size() << " inputs, "
		<< window.divisors.size() << " divisors)\n";
    }    
    if (window.feasible_sets.empty()) {
      if (config.verbose) std::cout << "  No feasible resubstitution found\n";
      continue;
    }
    if (config.verbose) {
      std::cout << "  ✓ Found " << window.feasible_sets.size() << " feasible set(s)\n";
    }
    // For each feasible set, synthesize one circuit and store in FeasibleSet::synths
    for (auto& fs : window.feasible_sets) {
      // Build binary relation for this feasible set
      std::vector<std::vector<bool>> br;
      generate_relation(window.truth_tables, fs.divisor_indices, window.inputs.size(), br);

      // Try selected synthesis engine with gate budget = mffc_size - 1
      aigman* synthesized_aig = nullptr;
      if (config.use_mockturtle) {
        synthesized_aig = synthesize_circuit_mockturtle(br, window.mffc_size - 1);
      } else {
        synthesized_aig = synthesize_circuit(br, window.mffc_size - 1);
      }
      if (!synthesized_aig) {
        if (config.verbose) {
          std::cout << "  ✗ Synthesis failed for set {";
          for (size_t i = 0; i < fs.divisor_indices.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << fs.divisor_indices[i];
          }
          std::cout << "} within gate limit" << "\n";
        }
        continue;
      }
      int gain = window.mffc_size - synthesized_aig->nGates;
      assert(gain > 0 && "Synthesized candidate must be beneficial (gain > 0)");
      if (config.verbose) {
        std::cout << "  ✓ Synthesized set {";
        for (size_t i = 0; i < fs.divisor_indices.size(); i++) {
          if (i) std::cout << ", ";
          std::cout << fs.divisor_indices[i];
        }
        std::cout << "}: " << synthesized_aig->nGates << " gates, gain=" << gain << "\n";
      }
      fs.synths.push_back(synthesized_aig);
    }
  }
  
  // Insertion via heap over (window, feasible_set) candidates
  if (config.verbose) {
    std::cout << "\nProcessing candidates via gain-ordered heap...\n";
  }
  int successful_resubs = inserter_process_windows_heap(aig, windows, config.verbose);

  // Cleanup: delete any remaining synthesized AIGs to avoid leaks
  for (auto& win : windows) {
    for (auto& fs : win.feasible_sets) {
      for (auto*& s : fs.synths) {
        if (s) { delete s; s = nullptr; }
      }
      fs.synths.clear();
    }
  }
  
  // Final statistics
  auto end_time = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end_time - start_time);
  int final_gates = aig.nGates;
  // successful_resubs already computed
  if (config.show_stats || config.verbose) {
    std::cout << "\nResubstitution complete:\n";
    std::cout << "  Windows extracted: " << total_extracted << "\n";
    std::cout << "  Windows processed: " << windows.size() << "\n";
    std::cout << "  Successful resubstitutions: " << successful_resubs << "\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Initial gates: " << initial_gates << "\n";
    std::cout << "  Final gates: " << final_gates << "\n";
    int gate_change = final_gates - initial_gates;
    if (gate_change > 0) {
      std::cout << "  Gate change: +" << gate_change << " gates added\n";
    } else if (gate_change < 0) {
      std::cout << "  Gate change: " << (-gate_change) << " gates saved\n";
    } else {
      std::cout << "  Gate change: no change\n";
    }
  }
  
  // Write output if specified
  if (!config.output_file.empty()) {
    if (config.verbose) {
      std::cout << "Writing optimized AIG to " << config.output_file << "...\n";
    }
    aig.write(config.output_file.c_str());
  }
  
  return 0;
  
}
