#include "synthesis.hpp"

#include <iostream>
#include <array>
#include <cassert>
#include <limits>
#include <unordered_map>

#include <kissat_solver.hpp>
#include <synth.hpp>

#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <mockturtle/algorithms/node_resynthesis/akers.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/utils/tech_library.hpp>
#include <mockturtle/views/topo_view.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <kitty/npn.hpp>
#include <kitty/operations.hpp>

using namespace std;

namespace fresub {

  static aigman* convert_mockturtle_aig_to_aigman(mockturtle::aig_network const& result_ntk, int num_inputs) {
    aigman* result_aig = new aigman(num_inputs, 1);
    std::unordered_map<mockturtle::aig_network::node, int> node_map;
    node_map.reserve(result_ntk.size());
    int pi_count = 0;
    result_ntk.foreach_pi([&](auto const& n, auto i) {
      (void)i;
      if (pi_count < num_inputs) {
        node_map[n] = pi_count + 1;
        ++pi_count;
      } else {
        node_map[n] = 0;
      }
    });
    result_ntk.foreach_gate([&](auto const& n) {
      std::vector<int> fanin_lits;
      result_ntk.foreach_fanin(n, [&](auto const& s, auto i) {
        (void)i;
        int fanin_node = result_ntk.get_node(s);
        int fanin_lit = node_map[fanin_node] * 2 + (result_ntk.is_complemented(s) ? 1 : 0);
        fanin_lits.push_back(fanin_lit);
      });
      assert(fanin_lits.size() == 2);
      int new_node_id = result_aig->newgate(fanin_lits[0], fanin_lits[1]);
      node_map[n] = new_node_id;
    });
    result_ntk.foreach_po([&](auto const& s, auto i) {
      (void)i;
      auto fanin_node = result_ntk.get_node(s);
      int output_lit = node_map[fanin_node] * 2 + (result_ntk.is_complemented(s) ? 1 : 0);
      result_aig->vPos[0] = output_lit;
    });
    return result_aig;
  }

  // Convert truth tables to exopt binary relation format
  void generate_relation(const vector<vector<uint64_t>>& truth_tables, const vector<int>& selected_divisors, int num_inputs, vector<vector<bool>>& br) {
    // We compute target function in terms of selected divisors
    // br[divisor_pattern][target_value] = can this divisor pattern produce this target value?
    // Initialize with all true (everything is don't care initially)
    int num_selected = selected_divisors.size();
    int num_divisor_patterns = 1 << num_selected;
    int total_patterns = 1 << num_inputs;
    // Initialize br - all patterns are don't care initially
    br.clear();
    br.resize(num_divisor_patterns, vector<bool>(2, true));
    // For each input pattern, extract divisor values and target value
    for (int input_pattern = 0; input_pattern < total_patterns; input_pattern++) {
      int word_idx = input_pattern / 64;
      int bit_idx = input_pattern % 64;
      // Extract target value for this input pattern
      bool target_value = (truth_tables.back()[word_idx] >> bit_idx) & 1;
      // Extract selected divisor values for this input pattern
      int divisor_pattern = 0;
      for (int i = 0; i < num_selected; i++) {
	int divisor_idx = selected_divisors[i];
	bool divisor_value = (truth_tables[divisor_idx][word_idx] >> bit_idx) & 1;
	if (divisor_value) {
	  divisor_pattern |= (1 << i);
	}
      }
      // This divisor pattern cannot produce the opposite target value
      br[divisor_pattern][target_value ? 0 : 1] = false;
    }
  }
  
  aigman* synthesize_circuit(const vector<vector<bool>>& br, int max_gates) {
    // Create synthesis manager - pass NULL for sim since we don't use it
    SynthMan<KissatSolver> synth_man(br, nullptr);
    // Attempt synthesis
    aigman* aig = nullptr;
    for(int i = 0; !aig && i <= max_gates; i++) {
      aig = synth_man.Synth(i);
    }
    // Return synthesized AIG or nullptr if synthesis failed
    // Note: DON'T delete aig here - it's needed for insertion
    // The caller will handle cleanup
    return aig;
  }

  // Get or create static mockturtle library instance
  mockturtle::exact_library<mockturtle::aig_network, 4>& get_mockturtle_library() {
    static mockturtle::xag_npn_resynthesis<mockturtle::aig_network, mockturtle::aig_network, 
					   mockturtle::xag_npn_db_kind::aig_complete> aig_resyn;
    static mockturtle::exact_library_params param;
    static mockturtle::exact_library<mockturtle::aig_network, 4> lib(aig_resyn, param);
    return lib;
  }

  // Helper: extend a 2^n-bit pattern to 4-inputs by replication
  static uint16_t extend_to_4_inputs(uint16_t pattern, int num_inputs) {
    uint16_t ext = pattern;
    if (num_inputs < 4) {
      for (int missing = num_inputs; missing < 4; ++missing) {
        int shift_amount = 1 << missing; // 2^missing
        ext |= static_cast<uint16_t>(ext << shift_amount);
      }
    }
    return ext;
  }

  static aigman* synthesize_mockturtle_exact(
    kitty::static_truth_table<4> const& tt,
    kitty::static_truth_table<4> dc,
    int num_inputs,
    int max_gates) {
    // NPN canonicalization of the function (not DC-aware)
    auto npn_result = kitty::exact_npn_canonization(tt);
    auto canonical_tt = std::get<0>(npn_result);
    auto neg = std::get<1>(npn_result);          // bitmask: inputs [0..3], output at bit 4
    auto perm_arr = std::get<2>(npn_result);     // array mapping canonical input i -> original input index

    // Transform DC mask into the same canonical orientation
    // Apply input negations
    for (int i = 0; i < 4; ++i) {
      if ((neg >> i) & 1) {
        kitty::flip_inplace(dc, i);
      }
    }
    // Apply permutation: new var i = old var perm_arr[i]
    // We'll perform by successive swaps using a position tracker
    std::array<uint8_t, 4> position{0,1,2,3};
    for (int i = 0; i < 4; ++i) {
      // find current index k of original var perm_arr[i]
      int target_orig = perm_arr[i];
      int k = 0;
      for (; k < 4; ++k) {
        if (position[k] == target_orig) break;
      }
      if (k != i) {
        kitty::swap_inplace(dc, i, k);
        std::swap(position[i], position[k]);
      }
    }

    // Query exact library with DC lookup
    auto& lib = get_mockturtle_library();
    uint32_t phase = static_cast<uint32_t>(neg);
    std::vector<uint8_t> perm_vec{ static_cast<uint8_t>(perm_arr[0]), static_cast<uint8_t>(perm_arr[1]), static_cast<uint8_t>(perm_arr[2]), static_cast<uint8_t>(perm_arr[3]) };
    auto supergates = lib.get_supergates(canonical_tt, dc, phase, perm_vec);
    if (!(supergates && !supergates->empty())) {
      return nullptr;
    }

    // Pick best by estimated area within max_gates
    auto best_gate = supergates->end();
    for (auto it = supergates->begin(); it != supergates->end(); ++it) {
      int estimated_gates = static_cast<int>(std::ceil(it->area));
      if (estimated_gates <= max_gates) {
        if (best_gate == supergates->end() || it->area < best_gate->area) {
          best_gate = it;
        }
      }
    }
    if (best_gate == supergates->end()) {
      return nullptr;
    }

    // Build the network using the possibly-updated phase/perm from the DC lookup
    mockturtle::aig_network result_ntk;
    std::vector<mockturtle::aig_network::signal> pis;
    pis.reserve(4);
    for (int i = 0; i < 4; ++i) {
      pis.push_back(result_ntk.create_pi());
    }
    std::vector<mockturtle::aig_network::signal> permuted_pis(4);
    for (int i = 0; i < 4; ++i) {
      int orig_input = perm_vec[i];
      auto signal = pis[orig_input];
      if ((phase >> orig_input) & 1) {
        signal = result_ntk.create_not(signal);
      }
      permuted_pis[i] = signal;
    }
    const auto& db = lib.get_database();
    mockturtle::topo_view topo_db{db, best_gate->root};
    auto extracted_signals = mockturtle::cleanup_dangling(topo_db, result_ntk, permuted_pis.begin(), permuted_pis.end());
    auto output_signal = extracted_signals.front();
    bool output_negated = (phase >> 4) & 1;
    if (output_negated) {
      output_signal = result_ntk.create_not(output_signal);
    }
    result_ntk.create_po(output_signal);

    return convert_mockturtle_aig_to_aigman(result_ntk, num_inputs);
  }

  aigman* synthesize_circuit_mockturtle(const vector<vector<bool>>& br, int max_gates) {
    // Determine number of inputs from BR size using ceiling log2
    int br_size = br.size();
    int num_inputs = 0;
    while ((1 << num_inputs) < br_size) {
      num_inputs++;
    }
    // Build fixed function truth table and don't-care mask (1-bit = don't care)
    uint16_t func_bits = 0;
    uint16_t dc_bits = 0;
    for (int pattern = 0; pattern < br_size; ++pattern) {
      const bool can0 = br[pattern][0];
      const bool can1 = br[pattern][1];
      if (can0 && can1) {
        dc_bits |= static_cast<uint16_t>(1u << pattern);
      } else if (!can0 && can1) {
        func_bits |= static_cast<uint16_t>(1u << pattern);
      } else if (!can0 && !can1) {
        // Impossible pattern
        return nullptr;
      } else {
        // can0 && !can1 => bit stays 0
      }
    }

    uint16_t func_bits_ext = extend_to_4_inputs(func_bits, num_inputs);
    uint16_t dc_bits_ext   = extend_to_4_inputs(dc_bits, num_inputs);
    kitty::static_truth_table<4> tt, dc;
    kitty::create_from_words(tt, &func_bits_ext, &func_bits_ext + 1);
    kitty::create_from_words(dc, &dc_bits_ext, &dc_bits_ext + 1);

    return synthesize_mockturtle_exact(tt, dc, num_inputs, max_gates);
  }

  aigman* synthesize_circuit_mockturtle(kitty::dynamic_truth_table const& function, int max_gates) {
    if (function.num_vars() > 4u) {
      return nullptr;
    }

    uint16_t func_bits = 0;
    const auto num_patterns = 1u << function.num_vars();
    for (uint32_t pattern = 0; pattern < num_patterns; ++pattern) {
      if (kitty::get_bit(function, pattern)) {
        func_bits |= static_cast<uint16_t>(1u << pattern);
      }
    }

    uint16_t func_bits_ext = extend_to_4_inputs(func_bits, static_cast<int>(function.num_vars()));
    uint16_t dc_bits_ext = 0;
    kitty::static_truth_table<4> tt, dc;
    kitty::create_from_words(tt, &func_bits_ext, &func_bits_ext + 1);
    kitty::create_from_words(dc, &dc_bits_ext, &dc_bits_ext + 1);

    return synthesize_mockturtle_exact(tt, dc, static_cast<int>(function.num_vars()), max_gates);
  }

  aigman* synthesize_lut_function(kitty::dynamic_truth_table const& function) {
    if (function.num_vars() <= 4u) {
      return synthesize_circuit_mockturtle(function, std::numeric_limits<int>::max());
    }

    mockturtle::aig_network result_ntk;
    std::vector<mockturtle::aig_network::signal> pis;
    pis.reserve(function.num_vars());
    for (uint32_t i = 0; i < function.num_vars(); ++i) {
      pis.push_back(result_ntk.create_pi());
    }

    mockturtle::akers_resynthesis<mockturtle::aig_network> resyn;
    resyn(result_ntk, function, pis.begin(), pis.end(), [&](auto const& f) {
      result_ntk.create_po(f);
    });

    return convert_mockturtle_aig_to_aigman(result_ntk, static_cast<int>(function.num_vars()));
  }

}
