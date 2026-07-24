#pragma once

#include <vector>

#include <aig.hpp>
#include "window.hpp"

namespace fresub {

  // Process windows directly using a gain-ordered heap over feasible sets.
  // Returns number of applied resubstitutions.
  int inserter_process_windows_heap(aigman& aig, std::vector<Window>& windows, bool verbose = false);

  // Process LUT windows using removed LUT count as the gain metric.
  int inserter_process_lut_windows(aigman& aig, std::vector<Window>& windows, bool verbose = false);

} // namespace fresub
