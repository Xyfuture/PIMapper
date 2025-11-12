"""Test recursive grid search with allocation tree for 5 channels and 4096x12288 matrix.

This test demonstrates:
1. Running recursive grid search on a large matrix
2. Building an allocation tree during the search
3. Validating that the mapping matches the tree
4. Analyzing channel statistics
"""

import logging
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType
from pimapper.matrixmapper.strategy.recursive_grid_search import RecursiveGridSearchStrategy
from pimapper.matrixmapper.utils import (
    validate_mapping_matches_tree,
    print_mapping_tree_comparison,
    get_channel_statistics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_5_channels_4096x12288():
    """Test recursive grid search with 5 channels on a 4096x12288 matrix."""

    print("=" * 80)
    print("Test: Recursive Grid Search with Allocation Tree")
    print("Matrix: 4096x12288, Channels: 5")
    print("=" * 80)

    # Create hardware specification
    # Using realistic PIM channel specs
    channel_spec = PIMChannelSpec(
        compute_power=4,  # 100 GOPS
        shared_bandwidth=12.8,  # 50 GB/s shared bandwidth
        memory_bandwidth=1.0  # 1 GB/s memory bandwidth
    )

    # Create accelerator with 5 channels
    accel_spec = AcceleratorSpec(
        channel_count=5,
        channel_spec=channel_spec
    )

    accelerator = Accelerator.create_from_spec(accel_spec)

    print(f"\nAccelerator Configuration:")
    print(f"  Channels: {len(accelerator.channels)}")
    print(f"  Compute power per channel: {channel_spec.compute_power} GOPS")
    print(f"  Total compute power: {accelerator.total_compute_power_gops} GOPS")

    # Create matrix specification
    matrix = MatrixShape(
        rows=4096,
        cols=12288,
        batch_size=32,
        data_format=DataFormat(
            input_dtype=DataType.FP16,
            output_dtype=DataType.FP16,
            weight_dtype=DataType.FP16
        )
    )

    print(f"\nMatrix Configuration:")
    print(f"  Shape: {matrix.rows}x{matrix.cols}x{matrix.batch_size}")
    print(f"  Total elements: {matrix.rows * matrix.cols * matrix.batch_size:,}")
    print(f"  Data format: {matrix.data_format.input_dtype.name}")

    # Create strategy with limited search space for faster execution
    strategy = RecursiveGridSearchStrategy(
        num_split_row_candidates=[1, 2,3, 5,4, 8],
        num_split_col_candidates=[1, 2, 3,5, 4, 8],
        max_iterations=2,
        enable_fallback_splits=False
    )

    print(f"\nStrategy Configuration:")
    print(f"  Row split candidates: {strategy.num_split_row_candidates}")
    print(f"  Col split candidates: {strategy.num_split_col_candidates}")
    print(f"  Max iterations: {strategy.max_iterations}")

    # Run the search
    print("\n" + "=" * 80)
    print("Running recursive grid search...")
    print("=" * 80)

    result = strategy.find_optimal_mapping(
        matrix_shape=matrix,
        accelerator=accelerator,
        current_iteration=0
    )

    if result is None:
        print("\nERROR: No valid mapping found!")
        return False

    print("\n" + "=" * 80)
    print("Search completed successfully!")
    print("=" * 80)

    # Print results
    print(f"\nMapping Results:")
    print(f"  Latency: {result.latency:.4f} ms")
    print(f"  Compute utilization: {result.get_compute_utilization():.2%}")

    # Check if tree was created
    if result.allocation_tree is None:
        print("\nWARNING: No allocation tree was created!")
        return False

    print(f"\nAllocation Tree:")
    print(f"  Tree created: Yes")
    print(f"  Root node: {result.allocation_tree.root.rows}x{result.allocation_tree.root.cols}x{result.allocation_tree.root.batch_size}")
    print(f"  Grid split: {result.allocation_tree.root.num_split_row}x{result.allocation_tree.root.num_split_col}")
    print(f"  Total tiles: {result.allocation_tree.root.total_tiles()}")

    # Validate tree structure
    print("\n" + "=" * 80)
    print("Validating tree structure...")
    print("=" * 80)

    tree_valid = result.allocation_tree.validate(check_allocations=True)
    print(f"Tree structure valid: {tree_valid}")

    if not tree_valid:
        print("ERROR: Tree validation failed!")
        return False

    # Validate mapping matches tree
    print("\n" + "=" * 80)
    print("Validating mapping matches tree...")
    print("=" * 80)

    mapping_valid = validate_mapping_matches_tree(result.mapping, result.allocation_tree)
    print(f"Mapping matches tree: {mapping_valid}")

    if not mapping_valid:
        print("ERROR: Mapping does not match tree!")
        return False

    # Print tree structure
    print("\n" + "=" * 80)
    print("Tree Structure:")
    print("=" * 80)
    result.allocation_tree.print_tree()

    # Print mapping vs tree comparison
    print("\n")
    print_mapping_tree_comparison(result.mapping, result.allocation_tree)

    # Get and print channel statistics
    print("\n" + "=" * 80)
    print("Channel Statistics:")
    print("=" * 80)

    stats = get_channel_statistics(result.mapping, result.allocation_tree)

    for channel_id in sorted(stats.keys()):
        channel_stats = stats[channel_id]
        print(f"\n  {channel_id}:")
        print(f"    Number of tiles: {channel_stats['num_tiles']}")
        print(f"    Tile IDs: {channel_stats['tile_ids']}")
        print(f"    Total operations: {channel_stats['total_operations']:,}")
        print(f"    Tile shapes: {channel_stats['tile_shapes'][:3]}{'...' if len(channel_stats['tile_shapes']) > 3 else ''}")

    # Calculate load balance
    total_ops = sum(s['total_operations'] for s in stats.values())
    avg_ops = total_ops / len(stats)
    max_ops = max(s['total_operations'] for s in stats.values())
    min_ops = min(s['total_operations'] for s in stats.values())

    print(f"\n  Load Balance:")
    print(f"    Total operations: {total_ops:,}")
    print(f"    Average per channel: {avg_ops:,.0f}")
    print(f"    Max per channel: {max_ops:,}")
    print(f"    Min per channel: {min_ops:,}")
    print(f"    Load imbalance: {(max_ops - min_ops) / avg_ops:.2%}")

    print("\n" + "=" * 80)
    print("TEST PASSED: All validations successful!")
    print("=" * 80)

    return True


def test_smaller_matrix():
    """Test with a smaller matrix for quick verification."""

    print("\n\n" + "=" * 80)
    print("Test: Smaller Matrix (1024x3072, 5 channels)")
    print("=" * 80)

    # Create hardware specification
    channel_spec = PIMChannelSpec(
        compute_power=100.0,
        shared_bandwidth=50.0,
        memory_bandwidth=1.0
    )

    accel_spec = AcceleratorSpec(
        channel_count=5,
        channel_spec=channel_spec
    )

    accelerator = Accelerator.create_from_spec(accel_spec)

    # Create smaller matrix
    matrix = MatrixShape(
        rows=1024,
        cols=3072,
        batch_size=1,
        data_format=DataFormat(
            input_dtype=DataType.FP16,
            output_dtype=DataType.FP16,
            weight_dtype=DataType.FP16
        )
    )

    print(f"\nMatrix: {matrix.rows}x{matrix.cols}x{matrix.batch_size}")

    # Create strategy
    strategy = RecursiveGridSearchStrategy(
        num_split_row_candidates=[1, 2, 4],
        num_split_col_candidates=[1, 2, 4],
        max_iterations=2
    )

    # Run the search
    print("\nRunning search...")
    result = strategy.find_optimal_mapping(
        matrix_shape=matrix,
        accelerator=accelerator,
        current_iteration=0
    )

    if result is None:
        print("ERROR: No valid mapping found!")
        return False

    print(f"\nResults:")
    print(f"  Latency: {result.latency:.4f} ms")
    print(f"  Utilization: {result.get_compute_utilization():.2%}")

    # Validate
    if result.allocation_tree is None:
        print("WARNING: No allocation tree!")
        return False

    tree_valid = result.allocation_tree.validate(check_allocations=True)
    mapping_valid = validate_mapping_matches_tree(result.mapping, result.allocation_tree)

    print(f"  Tree valid: {tree_valid}")
    print(f"  Mapping valid: {mapping_valid}")

    if tree_valid and mapping_valid:
        print("\nTEST PASSED!")
        return True
    else:
        print("\nTEST FAILED!")
        return False


if __name__ == "__main__":
    # Run smaller test first for quick verification
    # print("Running smaller matrix test first...")
    # small_test_passed = test_smaller_matrix()

    # if not small_test_passed:
    #     print("\n\nSmaller test failed, skipping large matrix test.")
    #     exit(1)

    # Run the main test with large matrix
    print("\n\nRunning large matrix test...")
    large_test_passed = test_5_channels_4096x12288()

    if large_test_passed:
        print("\n\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        exit(0)
    else:
        print("\n\n" + "=" * 80)
        print("TESTS FAILED!")
        print("=" * 80)
        exit(1)
