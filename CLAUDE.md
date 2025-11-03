# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PiMapper is a Python framework for mapping neural network models to Processing-In-Memory (PIM) hardware architectures. It converts PyTorch models into computation graphs, optimizes them, maps matrix operations to hardware tiles, and simulates execution.

## Running Tests

```bash
# Run the full pipeline test (tests model tracing, normalization, and simplification)
python test_full_pipeline.py
```

This test validates the complete conversion pipeline from PyTorch model to optimized computation graph, verifying that shape information is preserved through all transformation stages.

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

**`NxComputationGraph`** (`base.py`): NetworkX-based directed graph representing computation flow
- Nodes store `Op` objects, edges represent data dependencies
- **Edge creation is automatic**: Edges are inferred from node references in `Op.args` and `Op.kwargs`
- Key methods:
  - `create_node(name, op)`: Creates node and automatically establishes edges from input references
  - `replace_uses(src, dst)`: Redirects all uses of src node to dst node
  - `replace_input(node, old_input, new_input)`: Replaces a specific input connection
  - `merge_nodes(keep, remove)`: Merges two nodes, combining their connections

**Important**: When creating nodes, the graph automatically extracts string references from `op.args` and `op.kwargs` to create edges. You don't manually add edges.

### Operations (`pimapper/core/graph/ops/`)

**`Op` base class** (`base.py`): Abstract base for all operations
- `op_type`: Class variable identifying the operation type
- `args` / `kwargs`: Operation arguments (may contain string references to other nodes)
- `input_ops`: OrderedDict mapping input Op objects to ResultFilter (describes which output to use)
- `results`: List of GraphTensor objects describing output shapes/dtypes
- `convert_from_torch(torch_op)`: Class method for converting from TorchFxOp (implemented by subclasses)

**Tensor Transformation System** (`base.py`):
- `GraphTensor`: Records tensor shape and dtype without storing actual values
- `TensorTransform`: Records tensor operations (slice, transpose, permute, view, reshape, squeeze, unsqueeze, flatten, expand, contiguous)
- `ResultFilter`: Selects specific outputs from an Op and applies a chain of TensorTransform operations
  - `index`: Which output from the results list
  - `transforms`: List of TensorTransform operations applied sequentially
  - Supports method chaining: `filter.add_transform(TensorTransform.transpose(0, 1)).add_transform(...)`

**Operation Types**:
- `native.py`: Native ops (MatMulOp, VectorAddOp, VectorMulOp, SiLUOp, SoftmaxOp, RMSNormOp)
  - Each implements `convert_from_torch()` to convert from torch operations
  - Registered automatically with NormalizeOpsPass via `NATIVE_OPS` list
- `torch_compat.py`: Torch-compatible ops for initial conversion (TorchPlaceholderOp, TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp, etc.)
- `fusionmatrix.py`: FusionMatrixOp for fusing multiple matrix operations with shared inputs

### Hardware Specs (`pimapper/core/`)
- `hwspec.py`:
  - `ComputeDieSpec`: Immutable die specification (compute_power, bandwidth, memory_bandwidth)
  - `Chip` and `ComputeDie`: Runtime hardware entities
- `matrixspec.py`:
  - `MatrixShape`: Matrix dimensions with data format
  - `Tile`: Submatrix with dimensions (no position)
  - `Mapping`: Bidirectional mapping between tiles and compute dies

### Model Configuration (`pimapper/model/`)

**`ModelConfig`** (`base.py`): Dataclass for LLM architecture configuration
- Fields: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `vocab_size`, `model_type`
- Model cards stored as JSON in `pimapper/model/model_cards/` (e.g., `Meta-Llama-3-8B.json`)
- `load_model_config(card_path)`: Loads ModelConfig from JSON file
- `initialize_module(config)`: Creates a LLaMALayer module for tracing

**`LLaMALayer`** (`base.py`): Single transformer layer implementation
- Includes attention (wq, wk, wv, wo), FFN (w1, w2, w3), RMSNorm, and RotaryPositionEmbedding
- Used as the tracing target in `build_computation_graph()`

## Key Workflows

### Converting a PyTorch Model
```python
from pimapper.modelmapper.converter import build_computation_graph

# Full pipeline: trace → convert → normalize → simplify
fx_graph, comp_graph = build_computation_graph(
    card_path="pimapper/model/model_cards/Meta-Llama-3-8B.json",
    batch_size=1,
    seq_len=4,
    normalize=True,  # Convert torch ops to native ops
    simplify=True    # Remove non-essential ops (view/reshape/transpose)
)
```

**Pipeline stages**:
1. `trace_module()`: Uses torch.fx to trace the PyTorch module
2. `fx_to_computation_graph()`: Converts fx.Graph to NxComputationGraph with torch_compat ops
3. `NormalizeOpsPass`: Converts torch_compat ops to native ops (matmul, vector_add, silu, etc.)
4. `SimplifyGraphPass`: Removes view/reshape/transpose operations that don't affect computation

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

All passes inherit from `Pass` base class in `pimapper/modelmapper/passes/base.py`:

**`NormalizeOpsPass`** (`normalize_ops.py`):
- Converts torch_compat ops to native ops
- Uses a registration system: native ops register themselves via `NATIVE_OPS` list
- Each native op implements `convert_from_torch(torch_op)` to handle conversion
- Preserves `results` (shape/dtype info) during conversion

**`SimplifyGraphPass`** (`simplify.py`):
- Removes non-essential operations (view, reshape, transpose, permute, squeeze, unsqueeze, expand, contiguous)
- Preserves essential operations: native ops, call_module, core call_function/call_method, placeholder, output
- Uses `replace_uses()` to bypass removed nodes while maintaining graph connectivity

**`MatrixFusionPass`** (`matrix_fusion.py`):
- Fuses matrices with shared inputs using latest-start-time (LST) scheduling
- Identifies matrix ops (2D+ tensors), groups by shared inputs
- Computes LST via critical path method (CPM)
- Creates FusionMatrixOp nodes containing multiple fused operations

**`PassManager`** (`base.py`):
- Manages execution of multiple passes
- Two modes:
  - `"sequential"`: Run each pass once in order
  - `"iterative"`: Run passes repeatedly until convergence (no changes) or max iterations
- Collects statistics and metadata from each pass execution

Example:
```python
from pimapper.modelmapper.passes import PassManager, NormalizeOpsPass, SimplifyGraphPass

manager = PassManager(verbose=True)
manager.add_pass(NormalizeOpsPass())
manager.add_pass(SimplifyGraphPass())
stats = manager.run(graph, mode="sequential")
```

## Important Conventions

- **Node References**: In computation graphs, nodes reference inputs by string names (not object references). The graph automatically creates edges from these string references.
- **Op Types**: Use `op_type` class variable to identify operation types. This is a string like "matmul", "vector_add", "call_function", etc.
- **Shape Tracking**: Shape and dtype information is stored in `Op.results` as a list of `GraphTensor` objects. Always preserve this during transformations.
- **Automatic Edge Creation**: When calling `create_node()`, edges are automatically created by extracting string references from `op.args` and `op.kwargs`. Don't manually add edges.
- **Bandwidth Modes**: Dies support either separate input/output bandwidth OR shared bandwidth (mutually exclusive)
- **Data Formats**: `DataFormat` specifies input/output/weight dtypes (FP16, INT8, INT4, FP8)
