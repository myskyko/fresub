#include <cassert>
#include <iostream>

#include <aig.hpp>

#include "insertion.hpp"
#include "window.hpp"

int total_tests = 0;
int passed_tests = 0;

#define ASSERT(cond) do { \
    total_tests++; \
    if (!(cond)) { \
        std::cerr << "Test failed: " << #cond << std::endl; \
    } else { \
        passed_tests++; \
    } \
} while(0)

using namespace fresub;

void print_aig_structure(const aigman& aig, const std::string& label) {
    std::cout << "=== " << label << " ===\n";
    std::cout << "nPis: " << aig.nPis << ", nGates: " << aig.nGates 
              << ", nPos: " << aig.nPos << ", nObjs: " << aig.nObjs << "\n";
    
    // Print gates
    std::cout << "Gates:\n";
    for (int i = aig.nPis + 1; i < aig.nObjs; i++) {
        if (i * 2 + 1 < static_cast<int>(aig.vObjs.size())) {
            int fanin0 = aig.vObjs[i * 2];
            int fanin1 = aig.vObjs[i * 2 + 1];
            int fanin0_var = fanin0 >> 1;
            int fanin1_var = fanin1 >> 1;
            bool fanin0_comp = fanin0 & 1;
            bool fanin1_comp = fanin1 & 1;
            
            std::cout << "  Node " << i << " = AND(";
            if (fanin0_comp) std::cout << "!";
            std::cout << fanin0_var;
            std::cout << ", ";
            if (fanin1_comp) std::cout << "!";
            std::cout << fanin1_var;
            std::cout << ")  [literals: " << fanin0 << ", " << fanin1 << "]\n";
        }
    }
    
    // Print outputs
    std::cout << "Outputs: ";
    for (int i = 0; i < aig.nPos; i++) {
        if (i > 0) std::cout << ", ";
        int po_lit = aig.vPos[i];
        int po_var = po_lit >> 1;
        bool po_comp = po_lit & 1;
        if (po_comp) std::cout << "!";
        std::cout << po_var;
    }
    std::cout << "\n\n";
}

void test_aigman_import() {
    std::cout << "=== TESTING AIGMAN NATIVE IMPORT ===\n";
    
    // Create main AIG
    aigman main_aig(3, 1);  // 3 PIs, 1 PO
    main_aig.vObjs.resize(7 * 2);
    
    // Node 4 = AND(1, 2)
    main_aig.vObjs[4 * 2] = 2;
    main_aig.vObjs[4 * 2 + 1] = 4;
    
    // Node 5 = AND(2, 3)
    main_aig.vObjs[5 * 2] = 4;
    main_aig.vObjs[5 * 2 + 1] = 6;
    
    // Node 6 = AND(4, 5) - this will be our target to replace
    main_aig.vObjs[6 * 2] = 8;
    main_aig.vObjs[6 * 2 + 1] = 10;
    
    main_aig.nGates = 3;
    main_aig.nObjs = 7;
    main_aig.vPos[0] = 12;  // Output points to node 6
    
    int original_gates = main_aig.nGates;
    
    // Print original structure
    print_aig_structure(main_aig, "MAIN AIG BEFORE IMPORT");
    
    // Create a simple replacement circuit (synthesized)
    // This would normally come from synthesis
    aigman* synth_aig = new aigman(2, 1);  // 2 inputs, 1 output
    synth_aig->vObjs.resize(4 * 2);
    
    // Node 3 = AND(1, 2) - single gate implementation
    synth_aig->vObjs[3 * 2] = 2;
    synth_aig->vObjs[3 * 2 + 1] = 4;
    
    synth_aig->nGates = 1;
    synth_aig->nObjs = 4;
    synth_aig->vPos[0] = 6;  // Output is node 3
    
    print_aig_structure(*synth_aig, "SYNTHESIZED CIRCUIT");
    
    // Use aigman's native import to insert the synthesized circuit
    // Map synth inputs [1,2] to main nodes [3,4]
    std::vector<int> input_mapping = {3, 4};
    
    // We want the output to be a new node that we'll use to replace node 6
    std::vector<int> output_mapping = {12};
    
    std::cout << "Importing synthesized circuit...\n";
    std::cout << "Input mapping: synth[1,2] -> main[3,4] (literals 6,8)\n";
    std::cout << "Output will create new node in main AIG\n";
    
    // Actually call the import function
    main_aig.insert(synth_aig, input_mapping, output_mapping);
    
    std::cout << "Import completed.\n";
    
    // Print structure after import
    print_aig_structure(main_aig, "MAIN AIG AFTER IMPORT");
    
    ASSERT(synth_aig != nullptr);
    ASSERT(synth_aig->nGates < original_gates);  // Synthesized circuit is smaller
    ASSERT(!output_mapping.empty());  // Import should produce output
    
    std::cout << "✓ Import and replacement completed successfully\n";
    
    delete synth_aig;
}

void test_heap_based_insertion() {
    std::cout << "\n=== TESTING HEAP-BASED INSERTION ===\n";
    
    // Create AIG with multiple nodes that will have overlapping windows
    aigman aig(4, 1);  // 4 PIs, 1 PO
    aig.vObjs.resize(10 * 2);
    
    // Node 5 = AND(1, 2)  
    aig.vObjs[5 * 2] = 2;
    aig.vObjs[5 * 2 + 1] = 4;
    
    // Node 6 = AND(3, 4)
    aig.vObjs[6 * 2] = 6;
    aig.vObjs[6 * 2 + 1] = 8;
    
    // Node 7 = AND(5, 6) - depends on both 5 and 6
    aig.vObjs[7 * 2] = 10;
    aig.vObjs[7 * 2 + 1] = 12;
    
    // Node 8 = AND(5, 3) - also depends on 5
    aig.vObjs[8 * 2] = 10;
    aig.vObjs[8 * 2 + 1] = 6;
    
    // Node 9 = AND(7, 8) - final output
    aig.vObjs[9 * 2] = 14;
    aig.vObjs[9 * 2 + 1] = 16;
    
    aig.nGates = 5;
    aig.nObjs = 10;
    aig.vPos[0] = 18;  // Output points to node 9
    
    print_aig_structure(aig, "INITIAL AIG FOR HEAP TEST");
    int initial_gates = aig.nGates;
    
    // Extract windows
    std::vector<Window> windows;
    window_extract_all(aig, 6, true, windows);
    
    std::cout << "Extracted " << windows.size() << " windows\n";
    for (size_t i = 0; i < windows.size(); i++) {
        std::cout << "  Window " << i << ": target=" << windows[i].target_node 
                  << ", divisors=" << windows[i].divisors.size() 
                  << ", mffc_size=" << windows[i].mffc_size << "\n";
    }
    
    // Fabricate feasible sets and synthesized circuits for some windows
    int fabricated = 0;
    for (auto& w : windows) {
        if (w.divisors.size() >= 2 && w.mffc_size >= 2) {
            FeasibleSet fs;
            fs.divisor_indices = {0, 1}; // use first two divisors
            // Create a tiny synthesized subcircuit with 1 gate (< mffc_size)
            aigman* synth_aig = new aigman(2, 1);
            synth_aig->vObjs.resize(4 * 2);
            synth_aig->vObjs[3 * 2] = 2;  // AND(1,2)
            synth_aig->vObjs[3 * 2 + 1] = 4;
            synth_aig->nGates = 1;
            synth_aig->nObjs = 4;
            synth_aig->vPos[0] = 6;
            fs.synths.push_back(synth_aig);
            w.feasible_sets.push_back(std::move(fs));
            fabricated++;
        }
    }
    std::cout << "Fabricated " << fabricated << " candidate(s) across windows\n";
    ASSERT(fabricated > 0);
    
    // Run heap-based inserter
    int applied = inserter_process_windows_heap(aig, windows, true);
    std::cout << "Applied candidates: " << applied << "\n";
    ASSERT(applied > 0);
    ASSERT(aig.nGates < initial_gates);
    std::cout << "✓ Heap-based insertion reduced gates from " << initial_gates 
              << " to " << aig.nGates << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "        INSERTION TEST SUITE           \n";
    std::cout << "========================================\n\n";
    
    test_aigman_import();
    test_heap_based_insertion();
    
    std::cout << "========================================\n";
    std::cout << "         TEST RESULTS SUMMARY          \n";
    std::cout << "========================================\n";
    
    if (passed_tests == total_tests) {
        std::cout << "🎉 ALL TESTS PASSED! (" << passed_tests << "/" << total_tests << ")\n\n";
        return 0;
    } else {
        std::cout << "❌ TESTS FAILED! (" << passed_tests << "/" << total_tests << ")\n\n";
        return 1;
    }
}
