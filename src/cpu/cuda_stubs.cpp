#include <stdexcept>
#include <vector>

#include "window.hpp"

namespace fresub {

void feasibility_check_cuda(std::vector<Window>::iterator begin, std::vector<Window>::iterator end) {
  (void)begin;
  (void)end;
  throw std::runtime_error("CUDA support is disabled in this build");
}

void feasibility_check_cuda_all(std::vector<Window>::iterator begin, std::vector<Window>::iterator end) {
  (void)begin;
  (void)end;
  throw std::runtime_error("CUDA support is disabled in this build");
}

} // namespace fresub
