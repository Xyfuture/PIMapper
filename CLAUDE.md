# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PiMapper is a Python framework for mapping neural network models to Processing-In-Memory (PIM) hardware architectures. It converts PyTorch models into computation graphs, optimizes them, maps matrix operations to hardware tiles, and simulates execution.

## Core Architecture

The system follows a three-stage pipeline:

1. **Model Mapping** (`pimapper/modelmapper/`): Converts PyTorch models to computation graphs
   - `converter.py`: Traces PyTorch modules using torch.fx, converts to NxComputationGraph
   - `passes/`: Graph optimization passes (normalize ops, simplify, matrix fusion)
   - Key function: `build_computation_graph()` - full pipeline from model config to optimized graph

2. **Matrix Mapping** (`pimapper/matrixmapper/`): Maps matrix operations to hardware tiles
   - `strategy/trivial.py`: Grid-based tiling with round-robin die assignment
   - `strategy/h2llm_mapping.py`: H2LLM-specific mapping strategy
   - `strategy/agent_grid_search.py`: Agent-based grid search optimization

3. **Simulation** (`pimapper/sim/`): Discrete-event simulation of hardware execution
   - `sim_engine.py`: SimChip and SimComputeDie classes using Desim framework
   - Models input/compute/output pipelines with bandwidth and compute constraints
   - Generates Perfetto trace files for visualization

## Core Data Structures

### Computation Graph (`pimapper/core/graph/`)
- `base.py`: `NxComputationGraph` - NetworkX-based directed graph
  - Nodes store `Op` objects with metadata
  - Edges represent data dependencies
  - Methods: `create_node()`, `replace_uses()`, `replace_input()`, `merge_nodes()`

### Operations (`pimapper/core/graph/ops/`)
- `base.py`: `Op` abstract base class with `OpMetadata`
- `native.py`: Native ops (MatMulOp, VectorAddOp, VectorMulOp, SiLUOp, SoftmaxOp, RMSNormOp)
- `torch_compat.py`: Torch-compatible ops for initial conversion
- `fusionmatrix.py`: FusionMatrix op for fusing multiple matrix operations

### Hardware Specs (`pimapper/core/`)
- `hwspec.py`:
  - `ComputeDieSpec`: Immutable die specification (compute_power, bandwidth, memory_bandwidth)
  - `Chip` and `ComputeDie`: Runtime hardware entities
- `matrixspec.py`:
  - `MatrixShape`: Matrix dimensions with data format
  - `Tile`: Submatrix with dimensions (no position)
  - `Mapping`: Bidirectional mapping between tiles and compute dies

## Key Workflows

### Converting a PyTorch Model
```python
from pimapper.modelmapper.converter import build_computation_graph

fx_graph, comp_graph = build_computation_graph(
    card_path="model_config.json",
    normalize=True,  # Convert torch ops to native ops
    simplify=True    # Remove non-essential ops
)
```

### Creating a Hardware Mapping
```python
from pimapper.matrixmapper import TrivialTilingStrategy
from pimapper.core.hwspec import ChipSpec, ComputeDieSpec, Chip
from pimapper.core.matrixspec import MatrixShape

die_spec = ComputeDieSpec(compute_power=100, shared_bandwidth=50, memory_bandwidth=1.0)
chip_spec = ChipSpec(die_count=4, die_spec=die_spec)
chip = Chip.create_from_spec(chip_spec)

matrix = MatrixShape(rows=1024, cols=1024)
strategy = TrivialTilingStrategy()
mapping = strategy.create_balanced_mapping(matrix, chip)
```

### Running Simulation
```python
from pimapper.sim.sim_engine import simulate

cycles = simulate(chip, mapping, save_trace=True, trace_filename="trace.json")
```

## Graph Optimization Passes

Passes inherit from `Pass` base class in `pimapper/modelmapper/passes/base.py`:
- `NormalizeOpsPass`: Converts torch_compat ops to native ops
- `SimplifyGraphPass`: Removes view/reshape operations
- `MatrixFusionPass`: Fuses matrices with shared inputs using latest-start-time scheduling

Use `PassManager` to chain passes with sequential or iterative execution.

## Important Conventions

- **Node References**: In computation graphs, nodes reference inputs by string names (not object references)
- **Op Types**: Use `op_type` class variable to identify operation types
- **Metadata**: Store shape, dtype, and custom info in `OpMetadata`
- **Bandwidth Modes**: Dies support either separate input/output bandwidth OR shared bandwidth (mutually exclusive)
- **Data Formats**: `DataFormat` specifies input/output/weight dtypes (FP16, INT8, INT4, FP8)

## Testing

No test files found in the repository. When adding tests, follow Python conventions with `test_*.py` files.

## Code Generation

The `pimapper/codegen.py` file exists but is currently empty - this is likely for future code generation features.
