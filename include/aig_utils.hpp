#pragma once

#include <unordered_set>
#include <string>
#include <vector>

#include <aig.hpp>

namespace fresub {

// --- Literal helpers ---
inline int lit2var(int lit) { return lit >> 1; }
inline bool is_complemented(int lit) { return (lit & 1) != 0; }
inline int var2lit(int var, bool comp = false) { return (var << 1) | (comp ? 1 : 0); }

// Node accessibility helper (alive and in range)
bool is_node_accessible(const aigman& aig, int node);

// Compute the MFFC (maximum fanout-free cone) using a dereference counter array.
// - Assumes `deref` entries are all 0 on entry; the function will restore all
//   touched entries back to 0 before returning.
// - Returns the set of node IDs that belong to the MFFC, including the root.
// - Asserts `root` is not a PI.
std::unordered_set<int> compute_mffc(aigman& aig, int root, std::vector<int>& deref);

// Compute MFFC while excluding specific divisor nodes (and implicitly their TFI)
// by priming their deref counts to 1. The function will restore those entries
// back to 0 before returning.
std::unordered_set<int> compute_mffc_excluding_divisors(
  aigman& aig,
  int root,
  std::vector<int>& deref,
  const std::vector<int>& divisors_to_exclude);

// Compute transitive fanout from root, bounded to nodes in window_nodes.
std::unordered_set<int> compute_tfo_in_window(
  aigman& aig,
  int root,
  const std::vector<int>& window_nodes);

// Debug-print the full AIG structure (PIs, gates, POs) with a label.
void print_aig(const aigman& aig, const std::string& label = "AIG");

// Reads a BLIF LUT network, decomposes each LUT into AIG, and records the AIG
// root node for each original LUT in aig.vCoverRoots.
bool read_blif_as_cover_aig(
  const std::string& filename,
  aigman& aig,
  bool verbose = false);

} // namespace fresub
