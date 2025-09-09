# Fresub - Resubstitution Prototype

A prototype circuit optimization tool that performs functional resubstitution on And-Inverter Graphs (AIGs). This is an experimental implementation exploring window-based resubstitution with GPU acceleration.

## Features

- **Window-based resubstitution**: Extracts optimization windows using cut enumeration
- **Feasibility checking**: Finds feasible divisor combinations for circuit resubstitution
- **Multiple synthesis backends**: 
  - SAT-based synthesis using exopt
  - Library-based synthesis using mockturtle (default)
- **GPU acceleration**: CUDA implementations for parallel feasibility checking
- **Flexible feasibility options**:
  - CPU: MIN-SIZE mode (default) or ALL combinations
  - CUDA (first): GPU implementation finding first feasible solution
  - CUDA (all): GPU implementation finding all feasible sets

## Building

### Prerequisites

- CMake 3.18+
- CUDA toolkit (for GPU acceleration)
- C++17 compatible compiler
- Git (for submodules)

### Build Instructions

```bash
git clone <repository-url>
cd fresub
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Note**: CUDA compilation is enabled by default. CMakeLists.txt sets `CMAKE_CUDA_ARCHITECTURES=75` for modern GPUs. To target a different GPU architecture, set the `CMAKE_CUDA_ARCHITECTURES` environment variable before running cmake (e.g., `CMAKE_CUDA_ARCHITECTURES=80` for A100 or RTX 30xx series).

## Usage

### Basic Usage

```bash
./fresub input.aig [output.aig]
```

### Command Line Options

- `-c <size>`: Maximum cut size for window extraction (default: 4)
- `-k <K>`: Process K randomly selected windows (default: 100)
- `--seed <N>`: RNG seed for `-k` sampling (default: 42)
- `-v`: Verbose output showing detailed optimization process
- `-s`: Show statistics summary
- `--exopt`: Use SAT-based synthesis (exopt)
- `--mockturtle`: Use library-based synthesis (mockturtle, default)
- `--cuda`: Use GPU acceleration (finds first feasible solution per window)
- `--cuda-all`: Use GPU acceleration (finds all feasible solutions per window)
- `--feas-all`: CPU feasibility ALL mode (default is MIN-SIZE)

### Examples

```bash
# Basic optimization with verbose output
./fresub -v circuit.aig optimized.aig

# Use GPU acceleration with all feasible sets
./fresub --cuda-all -v circuit.aig optimized.aig

# Use SAT-based synthesis with statistics
./fresub --exopt -s circuit.aig optimized.aig

# Process a random batch of 200 windows
./fresub -k 200 --seed 123 -v circuit.aig optimized.aig
```

## Algorithm Overview

The prototype implements a basic window-based resubstitution algorithm:

1. **Window Extraction**: Uses cut enumeration to identify optimization windows
2. **Truth Table Computation**: Computes truth tables for window nodes and divisors  
3. **Feasibility Checking**: Finds combinations that can implement the target function
4. **Logic Synthesis**: Synthesizes replacement circuits using chosen combinations
5. **Circuit Insertion**: Applies optimizations while avoiding structural conflicts

## Implementation Details

### Feasibility Checking Variants

- **CPU Implementation**: Sequential search supporting MIN-SIZE or ALL enumeration
- **CUDA (first)**: Parallel GPU kernel optimized for finding first feasible solution
- **CUDA (all)**: GPU implementation using 4D indexing to find all feasible sets

### Synthesis Backends

- **mockturtle**: Uses precomputed optimal 4-input circuits with NPN canonicalization
- **exopt**: SAT-based synthesis for exact optimization within gate limits

### File Structure

```
src/
├── cpu/
│   ├── main.cpp           # Main application
│   ├── feasibility.cpp    # CPU feasibility checking
│   ├── synthesis.cpp      # Logic synthesis implementations
│   ├── simulation.cpp     # Truth table computation
│   ├── window.cpp         # Window extraction
│   └── insertion.cpp      # Circuit modification
├── cuda/
│   ├── resub_kernels.cu       # Original CUDA implementation (first solution)
│   └── resub_kernels_all.cu   # Advanced CUDA implementation (all solutions)
include/
└── *.hpp                  # Header files
```

## Testing

Run basic tests:

```bash
# Test with sample benchmark
./fresub -v ../benchmarks/mul2.aig

# Compare CPU vs GPU results
./fresub -v ../benchmarks/mul2.aig > cpu_results.txt
./fresub --cuda-all -v ../benchmarks/mul2.aig > gpu_results.txt
diff cpu_results.txt gpu_results.txt
```
