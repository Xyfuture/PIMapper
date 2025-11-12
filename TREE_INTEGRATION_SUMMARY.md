# Matrix Allocation Tree Integration Summary

## Overview

Successfully integrated a tree-based structure into the recursive grid search algorithm to track precise tile allocations and positions during matrix mapping.

## What Was Implemented

### 1. Core Tree Structure (`pimapper/matrixmapper/utils/matrix_allocation_tree.py`)

**Node Types:**
- **`InternalNode`**: Represents a matrix to be split
  - Stores matrix dimensions (rows, cols, batch_size)
  - Grid split configuration (num_split_row × num_split_col)
  - Source tile IDs and grid shape for non-root nodes
  - Can have 1 leaf child + 0-2 internal children

- **`LeafNode`**: Represents tiles assigned to channels
  - Contains tile allocations with IDs, channel assignments, and grid positions
  - Supports grouping tiles by channel

- **`TileAllocation`**: Individual tile assignment record
  - Tile ID (row-major order)
  - Channel ID
  - Grid position (row_idx, col_idx)

**Key Features:**
- Two-pass construction:
  1. **First pass**: Build tree structure with tile counts
  2. **Second pass**: Assign tile IDs in row-major order
- Tree validation (structure and allocations)
- Helper functions for tile position mapping

### 2. Integration with RecursiveGridSearchStrategy

**Modified Files:**
- `pimapper/matrixmapper/strategy/recursive_grid_search.py`
- `pimapper/core/utils.py` (added `allocation_tree` field to `MappingResult`)

**Key Changes:**
- `_evaluate_split_configuration` now builds allocation tree during recursion
- Tree is constructed alongside mapping generation
- Root node creates the tree, child nodes extend it
- Tile IDs are assigned after tree construction completes
- Tree is validated and attached to `MappingResult`

**New Helper Method:**
- `_calculate_tail_grid_info()`: Calculates grid information for tail regions
  - Returns (num_tiles, grid_shape, num_split_row, num_split_col) for each tail region
  - Handles 3 cases: partial row only, complete rows only, partial + complete rows

### 3. Validation Utilities (`pimapper/matrixmapper/utils/validation.py`)

**Functions:**
- `validate_mapping_matches_tree()`: Verifies mapping consistency with tree
  - Checks channel presence in both mapping and tree
  - Validates tile counts per channel
  - Verifies total tiles cover the matrix

- `print_mapping_tree_comparison()`: Debug visualization of mapping vs tree

- `get_channel_statistics()`: Analyzes channel utilization
  - Tile counts per channel
  - Total operations per channel
  - Tile shapes and IDs

### 4. Test Suite

**Test Files:**
- `test_matrix_allocation_tree.py`: Unit tests for tree structure
  - Simple allocation (no recursion)
  - Allocation with tail region
  - Allocation with two tail regions
  - Complex recursive allocation

- `test_recursive_grid_search_with_tree.py`: Integration test
  - **Small matrix**: 1024×3072 with 5 channels
  - **Large matrix**: 4096×12288 with 5 channels
  - Validates tree structure
  - Validates mapping matches tree
  - Prints channel statistics and load balance

## Test Results

### Small Matrix Test (1024×3072, 5 channels)
- ✅ Tree structure valid
- ✅ Mapping matches tree
- Latency: 1645.0 ms
- Utilization: 0.38%

### Large Matrix Test (4096×12288, 5 channels)
- ✅ Tree structure valid
- ✅ Mapping matches tree
- Latency: 25453.0 ms
- Utilization: 0.40%
- Grid split: 1×4 (4 tiles total)
- Tile distribution:
  - channel_0: 1 tile (4096×3072)
  - channel_1: 1 tile (4096×3072)
  - channel_2: 1 tile (4096×3072)
  - channel_3: 1 tile (4096×3072)
  - channel_4: 0 tiles
- Load imbalance: 125% (due to 5 channels but only 4 tiles)

## Key Benefits

1. **Precise Tile Tracking**: Know exactly which tiles are assigned to which channels
2. **Position Information**: Track tile positions in the original matrix via grid coordinates
3. **Validation**: Verify mapping correctness by comparing with tree structure
4. **Debugging**: Visualize allocation decisions through tree structure
5. **Analysis**: Calculate load balance and channel utilization statistics

## Usage Example

```python
from pimapper.matrixmapper.strategy.recursive_grid_search import RecursiveGridSearchStrategy
from pimapper.matrixmapper.utils import validate_mapping_matches_tree, print_mapping_tree_comparison

# Create strategy and run search
strategy = RecursiveGridSearchStrategy(
    num_split_row_candidates=[1, 2, 4, 8],
    num_split_col_candidates=[1, 2, 4, 8],
    max_iterations=2
)

result = strategy.find_optimal_mapping(
    matrix_shape=matrix,
    accelerator=accelerator
)

# Access the allocation tree
if result.allocation_tree:
    # Validate tree structure
    tree_valid = result.allocation_tree.validate(check_allocations=True)

    # Validate mapping matches tree
    mapping_valid = validate_mapping_matches_tree(result.mapping, result.allocation_tree)

    # Print tree structure
    result.allocation_tree.print_tree()

    # Print comparison
    print_mapping_tree_comparison(result.mapping, result.allocation_tree)

    # Get channel statistics
    stats = get_channel_statistics(result.mapping, result.allocation_tree)
```

## File Structure

```
pimapper/
├── core/
│   └── utils.py                          # Modified: Added allocation_tree to MappingResult
├── matrixmapper/
│   ├── strategy/
│   │   └── recursive_grid_search.py      # Modified: Integrated tree construction
│   └── utils/
│       ├── __init__.py                   # New: Exports tree and validation utilities
│       ├── matrix_allocation_tree.py     # New: Core tree structure
│       └── validation.py                 # New: Validation utilities

test_matrix_allocation_tree.py            # New: Unit tests for tree
test_recursive_grid_search_with_tree.py   # New: Integration test
```

## Implementation Notes

1. **Tree Construction**: Tree is built during the recursive search process, not as a post-processing step
2. **Tile ID Assignment**: IDs are assigned in row-major order after tree structure is complete
3. **Internal Children Priority**: When assigning IDs, internal children get IDs first (they have shape constraints), then leaf nodes get remaining IDs via round-robin
4. **Validation Modes**: Tree validation supports two modes:
   - `check_allocations=False`: Validate structure only (before ID assignment)
   - `check_allocations=True`: Validate structure and allocations (after ID assignment)
5. **Backwards Compatibility**: Existing code continues to work; tree is optional in `MappingResult`

## Future Enhancements

Potential improvements:
1. Add tile position in matrix coordinates (not just grid coordinates)
2. Support for non-uniform tile sizes in tree visualization
3. Export tree to visualization formats (GraphViz, JSON)
4. Performance metrics per tile (compute time, memory usage)
5. Tree-based optimization hints for better load balancing
