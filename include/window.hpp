#pragma once

#include <cstdint>
#include <vector>

#include <aig.hpp>
#include <cut.hpp>

namespace fresub {

  struct FeasibleSet {
    std::vector<int> divisor_indices; // indices into window.divisors
    std::vector<aigman*> synths;      // synthesized subcircuits for this set
  };

  struct Window {
    int target_node;
    std::vector<int> inputs;     // Boundary input variables.
    std::vector<int> nodes;      // Internal AIG nodes used for simulation.
    std::vector<int> divisors;   // Candidate divisors excluding MFFC/TFO.
    int cut_id;                  // ID of the cut that generated this window
    int mffc_size;
    std::vector<std::vector<uint64_t>> truth_tables;
    std::vector<FeasibleSet> feasible_sets; // optional: enriched storage per feasible set
  };

  struct BfsWindowParams {
    int max_inputs = 8;
    int max_nodes = 64;
  };

  // Extract all windows using exopt's cut enumeration.
  void window_extract_all(aigman& aig, int max_cut_size, bool verbose, std::vector<Window>& windows);

  // Extract graph windows by breadth-first TFI expansion over AIG gates.
  void window_extract_aig_bfs(aigman& aig, const BfsWindowParams& ps, bool verbose, std::vector<Window>& windows);

  // Extract graph windows by breadth-first TFI expansion over original LUT/cover roots.
  void window_extract_lut_bfs(aigman& aig, const BfsWindowParams& ps, bool verbose, std::vector<Window>& windows);

} // namespace fresub
