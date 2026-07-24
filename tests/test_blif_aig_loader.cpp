#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include "aig_utils.hpp"
#include "window.hpp"

int main() {
  {
    const char* filename = "/tmp/fresub_test_cover_roots.blif";
    {
      std::ofstream out(filename);
      out << ".model top\n";
      out << ".inputs a b c\n";
      out << ".outputs y\n";
      out << ".names a b n1\n";
      out << "11 1\n";
      out << ".names n1 c y\n";
      out << "11 1\n";
      out << ".end\n";
    }

    aigman aig;
    const bool ok = fresub::read_blif_as_cover_aig(filename, aig, false);
    assert(ok);
    assert(aig.nPis == 3);
    assert(aig.nPos == 1);
    assert(aig.nGates > 0);
    assert(aig.vCoverRoots.size() == 2);
    for (int root : aig.vCoverRoots) {
      assert(root > aig.nPis);
      assert(root < aig.nObjs);
      assert(root < static_cast<int>(aig.vvCoverInputs.size()));
      assert(!aig.vvCoverInputs[root].empty());
      for (int input : aig.vvCoverInputs[root]) {
        assert(input > 0);
        assert(input < aig.nObjs);
      }
    }

    fresub::BfsWindowParams ps;
    ps.max_inputs = 4;
    std::vector<fresub::Window> windows;
    fresub::window_extract_lut_bfs(aig, ps, false, windows);
    assert(!windows.empty());
    for (const auto& window : windows) {
      assert(std::find(aig.vCoverRoots.begin(), aig.vCoverRoots.end(), window.target_node) != aig.vCoverRoots.end());
      assert(!window.inputs.empty());
      assert(!window.nodes.empty());
    }

    const char* out_filename = "/tmp/fresub_test_cover_roots_out.blif";
    aig.write_blif(out_filename);
    aigman reread;
    const bool reread_ok = fresub::read_blif_as_cover_aig(out_filename, reread, false);
    assert(reread_ok);
    assert(reread.nPis == aig.nPis);
    assert(reread.nPos == aig.nPos);
    assert(reread.vCoverRoots.size() == aig.vCoverRoots.size());

    std::cout << "BLIF AIG loader test passed\n";
  }

  {
    const char* filename = "/tmp/fresub_test_cover_roots_5lut.blif";
    {
      std::ofstream out(filename);
      out << ".model top\n";
      out << ".inputs a b c d e\n";
      out << ".outputs y\n";
      out << ".names a b c d e y\n";
      out << "11111 1\n";
      out << ".end\n";
    }

    aigman aig;
    const bool ok = fresub::read_blif_as_cover_aig(filename, aig, false);
    assert(ok);
    assert(aig.nPis == 5);
    assert(aig.nPos == 1);
    assert(aig.nGates > 0);
    assert(aig.vCoverRoots.size() == 1);
    assert(aig.vCoverRoots[0] > aig.nPis);
    assert(aig.vCoverRoots[0] < aig.nObjs);
    assert(aig.vCoverRoots[0] < static_cast<int>(aig.vvCoverInputs.size()));
    assert(aig.vvCoverInputs[aig.vCoverRoots[0]].size() == 5);
  }

  std::cout << "BLIF AIG loader wide-LUT fallback test passed\n";

  {
    const char* filename = "/tmp/fresub_test_buffer_inverter.blif";
    {
      std::ofstream out(filename);
      out << ".model top\n";
      out << ".inputs a\n";
      out << ".outputs b inv\n";
      out << ".names a b\n";
      out << "1 1\n";
      out << ".names a inv\n";
      out << "1 0\n";
      out << ".end\n";
    }

    aigman aig;
    const bool ok = fresub::read_blif_as_cover_aig(filename, aig, false);
    assert(ok);
    assert(aig.nPis == 1);
    assert(aig.nPos == 2);
    assert(aig.nGates == 0);
    assert(aig.vCoverRoots.empty());
    assert(aig.vPos[0] == 2);
    assert(aig.vPos[1] == 3);
  }

  std::cout << "BLIF AIG loader buffer/inverter test passed\n";
  return 0;
}
