#include "window.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_set>
#include "aig_utils.hpp"
#include "graph.hpp"

namespace fresub {

namespace {

bool extract_bfs_tfi_window(const GraphView& graph, int target, const BfsWindowParams& ps, Window& window) {
  if (graph.fanins[target].empty()) {
    return false;
  }

  std::vector<bool> in_window(graph.nObjs, false);
  std::set<int> inputs;
  std::queue<int> queue;

  window = {};
  window.target_node = target;
  if (static_cast<int>(graph.fanins[target].size()) > ps.max_inputs) {
    return false;
  }
  in_window[target] = true;
  window.nodes.push_back(target);
  for (int fanin : graph.fanins[target]) {
    inputs.insert(fanin);
  }
  queue.push(target);

  while (!queue.empty()) {
    const int node = queue.front();
    queue.pop();

    for (int fanin : graph.fanins[node]) {
      if (in_window[fanin]) {
        continue;
      }
      if (!graph.fanins[fanin].empty() && static_cast<int>(window.nodes.size()) < ps.max_nodes) {
        assert(inputs.count(fanin));
        int new_input_count = static_cast<int>(inputs.size()) - 1;
        for (int fanin2 : graph.fanins[fanin]) {
          if (!in_window[fanin2] && !inputs.count(fanin2)) {
            ++new_input_count;
          }
        }
        if (new_input_count <= ps.max_inputs) {
          in_window[fanin] = true;
          window.nodes.push_back(fanin);
          inputs.erase(fanin);
          for (int fanin2 : graph.fanins[fanin]) {
            if (!in_window[fanin2]) {
              inputs.insert(fanin2);
            }
          }
          queue.push(fanin);
        }
      }
    }
  }

  std::queue<int> tfo_queue;
  for (int node : window.nodes) {
    tfo_queue.push(node);
  }
  while (!tfo_queue.empty() && static_cast<int>(window.nodes.size()) < ps.max_nodes) {
    const int node = tfo_queue.front();
    tfo_queue.pop();

    for (int fanout : graph.fanouts[node]) {
      if (in_window[fanout] || inputs.count(fanout)) {
        continue;
      }
      bool all_fanins_inside = true;
      for (int fanin : graph.fanins[fanout]) {
        if (!in_window[fanin]) {
          all_fanins_inside = false;
          break;
        }
      }
      if (all_fanins_inside) {
        in_window[fanout] = true;
        window.nodes.push_back(fanout);
        tfo_queue.push(fanout);
        if (static_cast<int>(window.nodes.size()) >= ps.max_nodes) {
          break;
        }
      }
    }
  }
  window.inputs.assign(inputs.begin(), inputs.end());
  std::sort(window.nodes.begin(), window.nodes.end());
  assert(!window.inputs.empty());
  return true;
}

} // namespace

void window_extract_all(aigman& aig, int max_cut_size, bool verbose, std::vector<Window>& windows) {
  assert(aig.fSorted);
  windows.clear();

  std::vector<std::vector<Cut>> cuts;

  if (verbose) std::cout << "Enumerating cuts using exopt...\n";
  CutEnumeration(aig, cuts, max_cut_size);

  if (verbose) std::cout << "Creating windows from cuts...\n";

  // Collect ALL cuts and assign global cut IDs
  std::vector<std::pair<int, Cut*>> all_cuts; // (target_node, cut)
  for (int target = aig.nPis + 1; target < aig.nObjs; target++) {
    for (auto& cut : cuts[target]) {
      if (cut.leaves.size() == 1 && cut.leaves[0] == target) {
        continue; // Skip trivial cut
      }
      assert(cut.leaves.size() <= static_cast<size_t>(max_cut_size));
      all_cuts.emplace_back(target, &cut);
    }
  }

  // Create lists for each node to store cut IDs
  std::vector<std::vector<int>> node_cut_lists(aig.nObjs);
  for (size_t cut_id = 0; cut_id < all_cuts.size(); cut_id++) {
    const Cut* cut = all_cuts[cut_id].second;
    for (int leaf : cut->leaves) {
      node_cut_lists[leaf].push_back(static_cast<int>(cut_id));
    }
  }

  // Propagate ALL cut IDs simultaneously in topological order
  std::vector<int> common_cuts; // Temporary storage
  for (int node = aig.nPis + 1; node < aig.nObjs; node++) {
    int fanin0 = lit2var(aig.vObjs[node * 2]);
    int fanin1 = lit2var(aig.vObjs[node * 2 + 1]);
    // Find intersection of cut IDs from both fanins
    common_cuts.clear();
    common_cuts.reserve(node_cut_lists[fanin0].size() + node_cut_lists[fanin1].size());
    std::set_intersection(node_cut_lists[fanin0].begin(), node_cut_lists[fanin0].end(),
                          node_cut_lists[fanin1].begin(), node_cut_lists[fanin1].end(),
                          std::back_inserter(common_cuts));
    // Merge the two sorted ranges
    std::vector<int> temp_result;
    temp_result.reserve(node_cut_lists[node].size() + common_cuts.size());
    std::set_union(node_cut_lists[node].begin(), node_cut_lists[node].end(),
                   common_cuts.begin(), common_cuts.end(),
                   std::back_inserter(temp_result));
    // Replace the old vector with the newly created, sorted union
    node_cut_lists[node] = std::move(temp_result);
  }

  // Create windows from propagated cut IDs
  windows.resize(all_cuts.size());
  for (size_t cut_id = 0; cut_id < all_cuts.size(); cut_id++) {
    windows[cut_id].target_node = all_cuts[cut_id].first;
    windows[cut_id].inputs = all_cuts[cut_id].second->leaves;
    windows[cut_id].cut_id = static_cast<int>(cut_id);
  }
  for (int i = 1; i < aig.nObjs; i++) {
    for (int cut_id : node_cut_lists[i]) {
      windows[cut_id].nodes.push_back(i);
    }
  }

  // Compute divisors = window nodes - MFFC(target) - TFO(target)
  std::vector<int> deref; // reuse across windows
  deref.assign(aig.nObjs, 0);
  for (auto& window : windows) {
    std::unordered_set<int> mffc = compute_mffc(aig, window.target_node, deref);
    std::unordered_set<int> tfo = compute_tfo_in_window(aig, window.target_node, window.nodes);
    for (int node : window.nodes) {
      if (mffc.find(node) == mffc.end() && tfo.find(node) == tfo.end()) {
        window.divisors.push_back(node);
      }
    }
    window.mffc_size = static_cast<int>(mffc.size());
  }
}

void window_extract_aig_bfs(aigman& aig, const BfsWindowParams& ps, bool verbose, std::vector<Window>& windows) {
  windows.clear();
  const auto graph = make_aig_graph(aig);
  for (int root : graph.roots) {
    Window window;
    if (!extract_bfs_tfi_window(graph, root, ps, window)) {
      continue;
    }

    window.cut_id = -1;

    std::vector<int> deref(aig.nObjs, 0);
    const auto mffc = compute_mffc(aig, window.target_node, deref);
    const auto tfo = compute_tfo_in_window(aig, window.target_node, window.nodes);
    for (int node : window.nodes) {
      if (mffc.find(node) == mffc.end() && tfo.find(node) == tfo.end()) {
        window.divisors.push_back(node);
      }
    }
    window.mffc_size = static_cast<int>(mffc.size());
    windows.push_back(std::move(window));
  }
  if (verbose) {
    std::cout << "Extracted " << windows.size() << " AIG BFS windows\n";
  }
}

void window_extract_lut_bfs(aigman& aig, const BfsWindowParams& ps, bool verbose, std::vector<Window>& windows) {
  windows.clear();
  const auto graph = make_lut_graph(aig);
  for (int root : graph.roots) {
    Window window;
    if (!extract_bfs_tfi_window(graph, root, ps, window)) {
      continue;
    }

    window.cut_id = -1;
    const std::vector<int> lut_nodes = window.nodes;

    std::vector<int> outputs;
    outputs.reserve(lut_nodes.size());
    for (int node : lut_nodes) {
      outputs.push_back(var2lit(node));
    }
    std::vector<int> gates;
    aig.getgates(gates, window.inputs, outputs);
    window.nodes = std::move(gates);

    std::vector<int> deref(aig.nObjs, 0);
    const auto mffc = compute_mffc(aig, window.target_node, deref);
    const auto tfo = compute_tfo_in_window(aig, window.target_node, window.nodes);
    for (int input : window.inputs) {
      window.divisors.push_back(input);
    }
    for (int node : lut_nodes) {
      if (node != window.target_node && mffc.find(node) == mffc.end() &&
          tfo.find(node) == tfo.end()) {
        window.divisors.push_back(node);
      }
    }
    window.mffc_size = static_cast<int>(mffc.size());
    windows.push_back(std::move(window));
  }
  if (verbose) {
    std::cout << "Extracted " << windows.size() << " LUT BFS windows\n";
  }
}

} // namespace fresub
