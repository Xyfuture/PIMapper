"""Test and demonstration of the matrix allocation tree.

This script demonstrates how to use the MatrixAllocationTree to track
tile allocations during recursive grid search.
"""

from pimapper.matrixmapper.utils import MatrixAllocationTree


def test_simple_allocation():
    """Test a simple allocation without recursion."""
    print("=" * 80)
    print("Test 1: Simple allocation (no recursion)")
    print("=" * 80)

    # Create a 4x4 matrix split into 2x2 grid (4 tiles total)
    # With 2 channels, each channel gets 2 tiles
    channel_ids = ["ch0", "ch1"]

    tree = MatrixAllocationTree.create_root(
        rows=4,
        cols=4,
        batch_size=1,
        num_split_row=2,
        num_split_col=2,
        channel_ids=channel_ids
    )

    # All 4 tiles go to the leaf node (no tail)
    tree.root.add_leaf_child(num_tiles=4)

    # Validate tree structure (before assignment)
    print(f"Tree structure valid: {tree.validate(check_allocations=False)}")

    # Assign tile IDs
    success = tree.assign_tile_ids()
    print(f"Tile ID assignment successful: {success}")

    # Validate tree with allocations
    print(f"Tree with allocations valid: {tree.validate(check_allocations=True)}")

    # Print tree structure
    print("\nTree structure:")
    tree.print_tree()

    # Get allocations by channel
    print("\nAllocations by channel:")
    allocations = tree.get_all_allocations()
    for channel_id, allocs in sorted(allocations.items()):
        tile_ids = [a.tile_id for a in allocs]
        positions = [a.tile_index_in_grid for a in allocs]
        print(f"  {channel_id}: tiles {tile_ids}, positions {positions}")

    print()


def test_allocation_with_tail():
    """Test allocation with tail region requiring recursion."""
    print("=" * 80)
    print("Test 2: Allocation with tail region")
    print("=" * 80)

    # Create a 6x6 matrix split into 2x3 grid (6 tiles total)
    # With 4 channels:
    # - Round-robin: 1 tile per channel = 4 tiles assigned
    # - Remaining: 2 tiles form a tail region
    channel_ids = ["ch0", "ch1", "ch2", "ch3"]

    tree = MatrixAllocationTree.create_root(
        rows=6,
        cols=6,
        batch_size=1,
        num_split_row=2,
        num_split_col=3,
        channel_ids=channel_ids
    )

    # 4 tiles go to leaf node (round-robin)
    tree.root.add_leaf_child(num_tiles=4)

    # 2 remaining tiles form a tail region (1 row x 2 cols in grid space)
    # This tail region will be recursively split
    tail_child = tree.root.add_internal_child(
        rows=3,  # Each tile is 3 rows (6/2)
        cols=4,  # 2 tiles of 2 cols each (6/3 * 2)
        batch_size=1,
        num_split_row=1,
        num_split_col=2,
        num_source_tiles=2,
        source_grid_shape=(1, 2)  # 1 row, 2 cols in parent grid
    )

    # The tail region has 2 tiles, all go to leaf
    tail_child.add_leaf_child(num_tiles=2)

    # Validate tree structure (before assignment)
    print(f"Tree structure valid: {tree.validate(check_allocations=False)}")

    # Assign tile IDs
    success = tree.assign_tile_ids()
    print(f"Tile ID assignment successful: {success}")

    # Validate tree with allocations
    print(f"Tree with allocations valid: {tree.validate(check_allocations=True)}")

    # Print tree structure
    print("\nTree structure:")
    tree.print_tree()

    # Get allocations by channel
    print("\nAllocations by channel:")
    allocations = tree.get_all_allocations()
    for channel_id, allocs in sorted(allocations.items()):
        tile_ids = [a.tile_id for a in allocs]
        positions = [a.tile_index_in_grid for a in allocs]
        print(f"  {channel_id}: tiles {tile_ids}, positions {positions}")

    print()


def test_allocation_with_two_tail_regions():
    """Test allocation with two tail regions."""
    print("=" * 80)
    print("Test 3: Allocation with two tail regions")
    print("=" * 80)

    # Create a 9x9 matrix split into 3x3 grid (9 tiles total)
    # With 2 channels:
    # - Round-robin: 4 tiles per channel = 8 tiles assigned
    # - Remaining: 1 tile forms a tail region
    channel_ids = ["ch0", "ch1"]

    tree = MatrixAllocationTree.create_root(
        rows=9,
        cols=9,
        batch_size=1,
        num_split_row=3,
        num_split_col=3,
        channel_ids=channel_ids
    )

    # 8 tiles go to leaf node (round-robin)
    tree.root.add_leaf_child(num_tiles=8)

    # 1 remaining tile forms a tail region
    tail_child = tree.root.add_internal_child(
        rows=3,  # Each tile is 3 rows (9/3)
        cols=3,  # Each tile is 3 cols (9/3)
        batch_size=1,
        num_split_row=1,
        num_split_col=1,
        num_source_tiles=1,
        source_grid_shape=(1, 1)  # 1 row, 1 col in parent grid
    )

    # The tail region has 1 tile, goes to leaf
    tail_child.add_leaf_child(num_tiles=1)

    # Validate tree structure (before assignment)
    print(f"Tree structure valid: {tree.validate(check_allocations=False)}")

    # Assign tile IDs
    success = tree.assign_tile_ids()
    print(f"Tile ID assignment successful: {success}")

    # Validate tree with allocations
    print(f"Tree with allocations valid: {tree.validate(check_allocations=True)}")

    # Print tree structure
    print("\nTree structure:")
    tree.print_tree()

    # Get allocations by channel
    print("\nAllocations by channel:")
    allocations = tree.get_all_allocations()
    for channel_id, allocs in sorted(allocations.items()):
        tile_ids = [a.tile_id for a in allocs]
        positions = [a.tile_index_in_grid for a in allocs]
        print(f"  {channel_id}: tiles {tile_ids}, positions {positions}")

    print()


def test_complex_recursive_allocation():
    """Test a more complex recursive allocation scenario."""
    print("=" * 80)
    print("Test 4: Complex recursive allocation")
    print("=" * 80)

    # Create a 12x12 matrix split into 4x4 grid (16 tiles total)
    # With 3 channels:
    # - Round-robin: 5 tiles per channel = 15 tiles assigned
    # - Remaining: 1 tile forms a tail region
    channel_ids = ["ch0", "ch1", "ch2"]

    tree = MatrixAllocationTree.create_root(
        rows=12,
        cols=12,
        batch_size=1,
        num_split_row=4,
        num_split_col=4,
        channel_ids=channel_ids
    )

    # 15 tiles go to leaf node (round-robin)
    tree.root.add_leaf_child(num_tiles=15)

    # 1 remaining tile forms a tail region
    tail_child = tree.root.add_internal_child(
        rows=3,  # Each tile is 3 rows (12/4)
        cols=3,  # Each tile is 3 cols (12/4)
        batch_size=1,
        num_split_row=1,
        num_split_col=1,
        num_source_tiles=1,
        source_grid_shape=(1, 1)
    )

    # The tail region has 1 tile, goes to leaf
    tail_child.add_leaf_child(num_tiles=1)

    # Validate tree structure (before assignment)
    print(f"Tree structure valid: {tree.validate(check_allocations=False)}")

    # Assign tile IDs
    success = tree.assign_tile_ids()
    print(f"Tile ID assignment successful: {success}")

    # Validate tree with allocations
    print(f"Tree with allocations valid: {tree.validate(check_allocations=True)}")

    # Print tree structure
    print("\nTree structure:")
    tree.print_tree()

    # Get allocations by channel
    print("\nAllocations by channel:")
    allocations = tree.get_all_allocations()
    for channel_id, allocs in sorted(allocations.items()):
        tile_ids = [a.tile_id for a in allocs]
        positions = [a.tile_index_in_grid for a in allocs]
        print(f"  {channel_id}: tiles {tile_ids}, positions {positions}")

    # Verify each channel gets tiles
    for channel_id in channel_ids:
        num_tiles = len(allocations[channel_id])
        print(f"  {channel_id} has {num_tiles} tiles")

    print()


if __name__ == "__main__":
    test_simple_allocation()
    test_allocation_with_tail()
    test_allocation_with_two_tail_regions()
    test_complex_recursive_allocation()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
