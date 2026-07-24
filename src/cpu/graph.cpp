#include "graph.hpp"

#include "aig_utils.hpp"

namespace fresub {

void graph_build_fanouts(GraphView& graph) {
  graph.fanouts.clear();
  graph.fanouts.resize(graph.nObjs);
  for (int node = 1; node < graph.nObjs; ++node) {
    for (int fanin : graph.fanins[node]) {
      graph.fanouts[fanin].push_back(node);
    }
  }
}

GraphView make_aig_graph(aigman const& aig) {
  GraphView graph;
  graph.nObjs = aig.nObjs;
  graph.fanins.resize(graph.nObjs);
  for (int node = aig.nPis + 1; node < aig.nObjs; ++node) {
    graph.fanins[node].push_back(lit2var(aig.vObjs[node * 2]));
    graph.fanins[node].push_back(lit2var(aig.vObjs[node * 2 + 1]));
    graph.roots.push_back(node);
  }
  graph_build_fanouts(graph);
  return graph;
}

GraphView make_lut_graph(aigman const& aig) {
  GraphView graph;
  graph.nObjs = aig.nObjs;
  graph.fanins.resize(graph.nObjs);
  for (int root : aig.vCoverRoots) {
    for (int input : aig.vvCoverInputs[root]) {
      graph.fanins[root].push_back(input);
    }
    graph.roots.push_back(root);
  }
  graph_build_fanouts(graph);
  return graph;
}

} // namespace fresub
