#include "aig_utils.hpp"

#include <iostream>
#include <cassert>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <kitty/operations.hpp>
#include <lorina/blif.hpp>
#include <mockturtle/io/blif_reader.hpp>
#include <mockturtle/networks/klut.hpp>

#include "synthesis.hpp"

namespace fresub {

bool is_node_accessible(const aigman& aig, int node) {
  if (node < 0 || node >= aig.nObjs) return false;
  return aig.vDeads.empty() || !aig.vDeads[node];
}

// Recursive helper for deref-based MFFC
static void mffc_deref_dfs(aigman& aig,
                           int n,
                           std::vector<int>& deref,
                           std::unordered_set<int>& cone,
                           std::vector<int>& touched) {
  int f0 = aig.vObjs[n * 2] >> 1;
  int f1 = aig.vObjs[n * 2 + 1] >> 1;
  int fis[2] = {f0, f1};
  for (int idx = 0; idx < 2; ++idx) {
    int fi = fis[idx];
    if (fi <= aig.nPis) continue; // stop at PIs
    if (deref[fi] == 0) touched.push_back(fi);
    ++deref[fi];
    int eff_ref = static_cast<int>(aig.vvFanouts[fi].size()) - deref[fi];
    if (eff_ref == 0) {
      cone.insert(fi);
      mffc_deref_dfs(aig, fi, deref, cone, touched);
    }
  }
}

std::unordered_set<int> compute_mffc(aigman& aig, int root, std::vector<int>& deref) {
  if (aig.vvFanouts.empty()) {
    aig.supportfanouts();
  }
  // Ensure deref has capacity; caller should have zero-initialized it
  if (static_cast<int>(deref.size()) < aig.nObjs) deref.resize(aig.nObjs);

  // Result container and touched list for restoring deref
  std::unordered_set<int> cone;
  cone.reserve(16);
  std::vector<int> touched;
  touched.reserve(32);

  // Root must be a gate and not pre-primed
  assert(root > aig.nPis);
  assert(deref[root] == 0);

  // Seed: pretend all fanouts of root are removed, so it enters the cone
  touched.push_back(root);
  deref[root] = static_cast<int>(aig.vvFanouts[root].size());
  cone.insert(root);

  // Recurse on fanins
  mffc_deref_dfs(aig, root, deref, cone, touched);

  // Restore deref to 0
  for (int t : touched) deref[t] = 0;
  return cone;
}

std::unordered_set<int> compute_mffc_excluding_divisors(
  aigman& aig,
  int root,
  std::vector<int>& deref,
  const std::vector<int>& divisors_to_exclude) {
  // Ensure deref has capacity and is zero-initialized for untouched entries
  if (static_cast<int>(deref.size()) < aig.nObjs) deref.resize(aig.nObjs);
  // Prime deref for divisor nodes to simulate a persistent external fanout
  // so they (and their TFI) never enter the MFFC during this run.
  for (int d : divisors_to_exclude) {
    // caller guarantees valid ids; no bounds checks for speed
    // use -1 so even after consuming all internal fanouts, eff_ref stays > 0
    deref[d] = -1;
  }

  auto cone = compute_mffc(aig, root, deref);

  // Restore divisor deref entries to 0
  for (int d : divisors_to_exclude) {
    deref[d] = 0;
  }
  return cone;
}

std::unordered_set<int> compute_tfo_in_window(
  aigman& aig,
  int root,
  const std::vector<int>& window_nodes) {
  std::unordered_set<int> tfo;
  std::unordered_set<int> window_set(window_nodes.begin(), window_nodes.end());
  if (aig.vvFanouts.empty()) {
    aig.supportfanouts();
  }
  std::queue<int> to_visit;
  to_visit.push(root);
  while (!to_visit.empty()) {
    int current = to_visit.front();
    to_visit.pop();
    if (tfo.find(current) == tfo.end()) {
      tfo.insert(current);
      for (int fanout : aig.vvFanouts[current]) {
        if (window_set.find(fanout) != window_set.end() &&
            tfo.find(fanout) == tfo.end()) {
          to_visit.push(fanout);
        }
      }
    }
  }
  return tfo;
}

void print_aig(const aigman& aig, const std::string& label) {
  std::cout << "=== " << label << " ===\n";
  std::cout << "nPis: " << aig.nPis << ", nGates: " << aig.nGates
            << ", nPos: " << aig.nPos << ", nObjs: " << aig.nObjs << "\n";

  std::cout << "PIs: ";
  for (int i = 1; i <= aig.nPis; ++i) {
    if (i > 1) std::cout << ", ";
    std::cout << i;
  }
  std::cout << "\n";

  std::cout << "Gates:\n";
  for (int i = aig.nPis + 1; i < aig.nObjs; ++i) {
    if (i * 2 + 1 < static_cast<int>(aig.vObjs.size())) {
      int f0 = aig.vObjs[i * 2];
      int f1 = aig.vObjs[i * 2 + 1];
      int v0 = f0 >> 1;
      int v1 = f1 >> 1;
      bool c0 = (f0 & 1) != 0;
      bool c1 = (f1 & 1) != 0;
      std::cout << "  Node " << i << " = AND(" << (c0 ? "!" : "") << v0
                << ", " << (c1 ? "!" : "") << v1 << ")  [lits: " << f0
                << ", " << f1 << "]\n";
    }
  }

  std::cout << "POs: ";
  for (int i = 0; i < aig.nPos; ++i) {
    if (i) std::cout << ", ";
    int lit = aig.vPos[i];
    int var = lit >> 1;
    bool comp = (lit & 1) != 0;
    if (comp) std::cout << "!";
    std::cout << var << " [lit: " << lit << "]";
  }
  std::cout << "\n\n";
}

bool read_blif_as_cover_aig(
  const std::string& filename,
  aigman& aig,
  bool verbose) {
  mockturtle::klut_network ntk;
  auto const read_result = lorina::read_blif(filename, mockturtle::blif_reader(ntk));
  if (read_result != lorina::return_code::success) {
    return false;
  }

  aig = aigman(static_cast<int>(ntk.num_pis()), static_cast<int>(ntk.num_pos()));

  std::unordered_map<mockturtle::klut_network::node, int> node_to_lit;
  node_to_lit.reserve(ntk.size());
  const auto const0_signal = ntk.get_constant(false);
  const auto const1_signal = ntk.get_constant(true);
  node_to_lit[ntk.get_node(const0_signal)] = ntk.is_complemented(const0_signal) ? 1 : 0;
  if (ntk.get_node(const1_signal) != ntk.get_node(const0_signal)) {
    node_to_lit[ntk.get_node(const1_signal)] = ntk.is_complemented(const1_signal) ? 0 : 1;
  }

  ntk.foreach_pi([&](auto const& n, auto i) {
    node_to_lit[n] = var2lit(static_cast<int>(i) + 1);
  });

  ntk.foreach_gate([&](auto const& n) {
    std::vector<int> inputs;
    auto function = ntk.node_function(n);
    int input_index = 0;
    ntk.foreach_fanin(n, [&](auto const& f) {
      const auto fanin = ntk.get_node(f);
      const int lit = node_to_lit.at(fanin) ^ (ntk.is_complemented(f) ? 1 : 0);
      inputs.push_back(lit2var(lit));
      if (is_complemented(lit)) {
        kitty::flip_inplace(function, input_index);
      }
      ++input_index;
    });

    if (function.num_vars() == 0u) {
      node_to_lit[n] = kitty::get_bit(function, 0) ? 1 : 0;
      return;
    }
    assert(function.num_vars() != 0u);
    if (function.num_vars() == 1u && inputs.size() == 1u) {
      const bool bit0 = kitty::get_bit(function, 0);
      const bool bit1 = kitty::get_bit(function, 1);
      if (!bit0 && bit1) {
        node_to_lit[n] = var2lit(inputs[0]);
        return;
      }
      if (bit0 && !bit1) {
        node_to_lit[n] = var2lit(inputs[0], true);
        return;
      }
    }

    aigman *lut_aig = synthesize_lut_function(function);
    if (!lut_aig) {
      throw std::runtime_error("failed to synthesize LUT while reading BLIF");
    }
    int output_lit = aig.append_cover(*lut_aig, inputs);
    delete lut_aig;

    node_to_lit[n] = output_lit;
  });

  ntk.foreach_po([&](auto const& f, auto i) {
    const auto node = ntk.get_node(f);
    aig.vPos[static_cast<size_t>(i)] = node_to_lit.at(node) ^ (ntk.is_complemented(f) ? 1 : 0);
  });

  aig.supportfanouts();
  aig.remove_dangling_nodes();
  aig.remove_dead_covers();

  if (verbose) {
    std::cout << "Read BLIF as AIG: " << aig.nPis << " PIs, " << aig.nPos
              << " POs, " << aig.nGates << " gates, " << aig.vCoverRoots.size()
              << " cover roots\n";
  }

  return true;
}

} // namespace fresub
