#include "insertion.hpp"

#include <iostream>
#include <cassert>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <kitty/bit_operations.hpp>
#include <kitty/dynamic_truth_table.hpp>
#include "aig_utils.hpp"
#include "synthesis.hpp"

namespace fresub {

  // use is_node_accessible from aig_utils.hpp

  // Internal heap item for gain-based processing
  struct HeapItem {
    int gain;
    int window_idx;
    int fs_idx;
    int synth_idx;
  };

  struct HeapCmp {
    bool operator()(HeapItem const& a, HeapItem const& b) const {
      return a.gain < b.gain; // max-heap by gain
    }
  };

  int inserter_process_windows_heap(aigman& aig, std::vector<Window>& windows, bool verbose) {
    if (verbose) {
      std::cout << "Building gain heap from windows and feasible sets...\n";
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> heap;

    // Build heap of all synthesized candidates; require positive gain
    for (size_t wi = 0; wi < windows.size(); ++wi) {
      auto& win = windows[wi];
      for (size_t fi = 0; fi < win.feasible_sets.size(); ++fi) {
        auto& fs = win.feasible_sets[fi];
        for (size_t si = 0; si < fs.synths.size(); ++si) {
          auto* synth = fs.synths[si];
          if (!synth) continue;
          int estimated_gain = win.mffc_size - synth->nGates;
          assert(estimated_gain > 0 && "Non-beneficial candidate should be filtered before insertion heap");
          heap.push(HeapItem{estimated_gain, static_cast<int>(wi), static_cast<int>(fi), static_cast<int>(si)});
        }
      }
    }

    int applied = 0;
    int skipped = 0;
    if (verbose) {
      std::cout << "Processing heap with " << heap.size() << " candidates...\n";
    }
    // Reusable deref buffer for MFFC computation
    std::vector<int> deref;
    while (!heap.empty()) {
      auto item = heap.top();
      heap.pop();

      auto& win = windows[item.window_idx];
      auto& fs = win.feasible_sets[item.fs_idx];
      if (item.synth_idx >= static_cast<int>(fs.synths.size())) continue;
      aigman* synth = fs.synths[item.synth_idx];
      if (!synth) continue; // may have been consumed/cleaned in a prior step

      // Validate target and divisors still exist and are acyclic
      if (!is_node_accessible(aig, win.target_node)) {
        skipped++;
        continue;
      }
      std::vector<int> selected_nodes;
      selected_nodes.reserve(fs.divisor_indices.size());
      for (int idx : fs.divisor_indices) {
        int node = win.divisors[idx];
        if (!is_node_accessible(aig, node)) { selected_nodes.clear(); break; }
        selected_nodes.push_back(node);
      }
      if (selected_nodes.empty() && !fs.divisor_indices.empty()) {
        skipped++;
        continue;
      }
      if (!selected_nodes.empty()) {
        std::vector<int> target_nodes = {win.target_node};
        if (aig.reach(target_nodes, selected_nodes)) {
          skipped++;
          continue;
        }
      }

      // Recompute current MFFC-based gain for the target node
      // Exclude selected divisors by priming their deref counts
      auto mffc_now = compute_mffc_excluding_divisors(aig, win.target_node, deref, selected_nodes);
      int current_gain = static_cast<int>(mffc_now.size()) - synth->nGates;
      if (current_gain <= 0) {
        // No longer beneficial after prior insertions
        skipped++;
        continue;
      }

      // Import synthesized circuit to replace target
      int gates_before = aig.nGates;
      std::vector<int> outputs = {win.target_node << 1};
      aig.insert(synth, selected_nodes, outputs);
      int actual_gain = gates_before - aig.nGates;
      if (verbose) {
        std::cout << "Applied candidate: target=" << win.target_node
                  << ", divs=" << selected_nodes.size()
                  << ", gates=" << synth->nGates
                  << ", gain=" << current_gain
                  << ", actual_gain=" << actual_gain << "\n";
      }
      // Note: actual_gain may exceed current_gain due to constant propagation and downstream simplifications
      assert(actual_gain >= current_gain);
      applied++;
    }

    if (verbose) {
      std::cout << "Heap processing complete: " << applied << " applied, " << skipped << " skipped\n";
    }
    return applied;
  }

  static int count_cover_roots(aigman const& aig, std::unordered_set<int> const& nodes) {
    int count = 0;
    for (int root : aig.vCoverRoots) {
      if (nodes.count(root)) {
        ++count;
      }
    }
    return count;
  }

  int inserter_process_lut_windows(aigman& aig, std::vector<Window>& windows, bool verbose) {
    if (verbose) {
      std::cout << "Building LUT gain heap from windows and feasible sets...\n";
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> heap;
    std::vector<int> deref;
    for (size_t wi = 0; wi < windows.size(); ++wi) {
      auto& win = windows[wi];
      for (size_t fi = 0; fi < win.feasible_sets.size(); ++fi) {
        auto& fs = win.feasible_sets[fi];
        std::vector<int> selected_nodes;
        selected_nodes.reserve(fs.divisor_indices.size());
        for (int idx : fs.divisor_indices) {
          selected_nodes.push_back(win.divisors[idx]);
        }
        const auto mffc = compute_mffc_excluding_divisors(aig, win.target_node, deref, selected_nodes);
        const int removed_luts = count_cover_roots(aig, mffc);
        const int gain = removed_luts - 1;
        if (gain <= 0) {
          continue;
        }
        heap.push(HeapItem{gain, static_cast<int>(wi), static_cast<int>(fi), -1});
      }
    }

    int applied = 0;
    int skipped = 0;
    if (verbose) {
      std::cout << "Processing LUT heap with " << heap.size() << " candidates...\n";
    }

    while (!heap.empty()) {
      auto item = heap.top();
      heap.pop();

      auto& win = windows[item.window_idx];
      auto& fs = win.feasible_sets[item.fs_idx];
      if (!is_node_accessible(aig, win.target_node)) {
        ++skipped;
        continue;
      }

      std::vector<int> selected_nodes;
      selected_nodes.reserve(fs.divisor_indices.size());
      for (int idx : fs.divisor_indices) {
        int node = win.divisors[idx];
        if (!is_node_accessible(aig, node)) { selected_nodes.clear(); break; }
        selected_nodes.push_back(node);
      }
      if (selected_nodes.empty() && !fs.divisor_indices.empty()) {
        ++skipped;
        continue;
      }
      if (!selected_nodes.empty()) {
        std::vector<int> target_nodes = {win.target_node};
        if (aig.reach(target_nodes, selected_nodes)) {
          ++skipped;
          continue;
        }
      }

      const auto mffc = compute_mffc_excluding_divisors(aig, win.target_node, deref, selected_nodes);
      const int removed_luts = count_cover_roots(aig, mffc);
      const int gain = removed_luts - 1;
      if (gain <= 0) {
        ++skipped;
        continue;
      }

      const int num_divisors = static_cast<int>(fs.divisor_indices.size());
      const int num_window_patterns = 1 << static_cast<int>(win.inputs.size());
      kitty::dynamic_truth_table function(num_divisors);
      for (int pattern = 0; pattern < num_window_patterns; pattern++) {
        int divisor_pattern = 0;
        for (int i = 0; i < num_divisors; i++) {
          const auto& tt = win.truth_tables[fs.divisor_indices[i]];
          if ((tt[pattern >> 6] >> (pattern & 63)) & 1) {
            divisor_pattern |= 1 << i;
          }
        }
        const auto& target = win.truth_tables.back();
        if ((target[pattern >> 6] >> (pattern & 63)) & 1) {
          kitty::set_bit(function, divisor_pattern);
        }
      }
      aigman* synth = synthesize_lut_function(function);
      if (!synth) {
        throw std::runtime_error("failed to synthesize LUT replacement");
      }
      const int new_lit = aig.append_cover(*synth, selected_nodes);
      delete synth;
      aig.vCoverRoots.erase(std::remove_if(aig.vCoverRoots.begin(), aig.vCoverRoots.end(),
                                           [&](int root) { return mffc.count(root); }),
                            aig.vCoverRoots.end());
      for (int root : mffc) {
        if (root < static_cast<int>(aig.vvCoverInputs.size())) {
          aig.vvCoverInputs[root].clear();
        }
      }
      aig.replacenode(win.target_node, new_lit, false);

      if (verbose) {
        std::cout << "Applied LUT candidate: target=" << win.target_node
                  << ", divs=" << selected_nodes.size()
                  << ", removed_luts=" << removed_luts
                  << ", gain=" << gain << "\n";
      }
      ++applied;
    }

    if (verbose) {
      std::cout << "LUT heap processing complete: " << applied << " applied, " << skipped << " skipped\n";
    }
    return applied;
  }

} // namespace fresub
