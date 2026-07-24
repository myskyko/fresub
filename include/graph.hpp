#pragma once

#include <vector>

#include <aig.hpp>

namespace fresub {

struct GraphView {
  int nObjs = 0;
  std::vector<std::vector<int>> fanins;
  std::vector<std::vector<int>> fanouts;
  std::vector<int> roots;
};

void graph_build_fanouts(GraphView& graph);

GraphView make_aig_graph(aigman const& aig);

GraphView make_lut_graph(aigman const& aig);

} // namespace fresub
