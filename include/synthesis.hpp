#pragma once

#include <cstdint>
#include <vector>

#include <aig.hpp>
#include <kitty/dynamic_truth_table.hpp>

namespace fresub {

  // Convert truth tables to exopt binary relation format
  void generate_relation(const std::vector<std::vector<uint64_t>>& truth_tables, const std::vector<int>& selected_divisors, int num_inputs, std::vector<std::vector<bool>>& br);
  
  // Synthesize optimal circuit from binary relation (exopt-based)
  // Returns synthesized aigman* or nullptr if synthesis fails
  aigman* synthesize_circuit(const std::vector<std::vector<bool>>& br, int max_gates);

  // Synthesize optimal circuit using mockturtle library lookup (4-input only)
  // Returns synthesized aigman* or nullptr if synthesis fails or exceeds max_gates
  aigman* synthesize_circuit_mockturtle(const std::vector<std::vector<bool>>& br, int max_gates);

  // Synthesize a complete truth table using mockturtle library lookup (up to 4 inputs).
  // Returns synthesized aigman* or nullptr if synthesis fails or exceeds max_gates.
  aigman* synthesize_circuit_mockturtle(kitty::dynamic_truth_table const& function, int max_gates);

  // Synthesize an AIG implementation for a complete LUT truth table.
  // Uses the exact/mockturtle library path for up to 4 inputs and a mockturtle
  // wide-function fallback otherwise.
  aigman* synthesize_lut_function(kitty::dynamic_truth_table const& function);

}
