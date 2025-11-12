"""Test tile ID hierarchy and uniqueness in matrix allocation tree.

This test file validates that:
1. Global tile IDs are unique across the entire tree
2. Local tile IDs are correctly assigned within each node (starting from 0)
3. Multi-level recursive structures maintain correct ID mappings

Original prompt:
采用方案2, 我要确保 id 不要重复, 也能知道层级内的id 排序, 然后看看这个改动会影响到
pimapper/matrixmapper/strategy/ 下面的哪些文件, 一起修复, 然后编写一个测试文件看一下
"""

from pimapper.matrixmapper.utils import MatrixAllocationTree


def test_global_tile_id_uniqueness():
    """Test that global tile IDs are unique across all levels of the tree."""
    print("=" * 80)
    print("Test 1: Global Tile ID Uniqueness")
    print("=" * 80)

    # Create a tree with 2 levels of recursion
    # Root: 3x3 grid (9 tiles), 3 channels
    # - Leaf: 6 tiles (2 per channel)
    # - Internal child: 3 tiles (1x3 grid)
    #   - Leaf: 3 tiles (1 per channel)

    channel_ids = ["ch0", "ch1", "ch2"]

    tree = MatrixAllocationTree.create_root(
        rows=9,
        cols=9,
        batch_size=1,
        num_split_row=3,
        num_split_col=3,
        channel_ids=channel_ids
    )

    # Root level: 6 tiles to leaf
    tree.root.add_leaf_child(num_tiles=6)

    # Root level: 3 tiles to internal child (tail region)
    tail_child = tree.root.add_internal_child(
        rows=3,
        cols=9,
        batch_size=1,
        num_split_row=1,
        num_split_col=3,
        num_source_tiles=3,
        source_grid_shape=(1, 3)
    )

    # Tail child: all 3 tiles to leaf
    tail_child.add_leaf_child(num_tiles=3)

    # Validate and assign IDs
    assert tree.validate(check_allocations=False), "Tree structure validation failed"

    success = tree.assign_tile_ids()
    assert success, "Tile ID assignment failed"

    assert tree.validate(check_allocations=True), "Tree validation with allocations failed"

    # Collect all global tile IDs
    all_allocations = tree.get_all_allocations()
    all_global_ids = []
    all_local_ids_by_level = {"root": [], "child": []}

    # Collect from root leaf
    if tree.root.leaf_child:
        for alloc in tree.root.leaf_child.allocations:
            all_global_ids.append(alloc.global_tile_id)
            all_local_ids_by_level["root"].append(alloc.tile_id)

    # Collect from child leaf
    for child in tree.root.internal_children:
        if child.leaf_child:
            for alloc in child.leaf_child.allocations:
                all_global_ids.append(alloc.global_tile_id)
                all_local_ids_by_level["child"].append(alloc.tile_id)

    # Test 1: Global IDs must be unique
    print(f"\nTotal tiles: {len(all_global_ids)}")
    print(f"Global IDs: {sorted(all_global_ids)}")
    print(f"Unique global IDs: {len(set(all_global_ids))}")

    assert len(all_global_ids) == len(set(all_global_ids)), \
        f"Global tile IDs are not unique! Found duplicates in {all_global_ids}"

    # Test 2: Global IDs should cover 0 to N-1
    expected_global_ids = set(range(9))
    actual_global_ids = set(all_global_ids)
    assert actual_global_ids == expected_global_ids, \
        f"Global IDs don't cover expected range. Expected {expected_global_ids}, got {actual_global_ids}"

    # Test 3: Local IDs at each level should start from 0
    print(f"\nRoot level local IDs: {sorted(all_local_ids_by_level['root'])}")
    print(f"Child level local IDs: {sorted(all_local_ids_by_level['child'])}")

    # Root level should have local IDs 0-5 (6 tiles)
    assert set(all_local_ids_by_level["root"]) == set(range(6)), \
        f"Root level local IDs incorrect: {all_local_ids_by_level['root']}"

    # Child level should have local IDs 0-2 (3 tiles)
    assert set(all_local_ids_by_level["child"]) == set(range(3)), \
        f"Child level local IDs incorrect: {all_local_ids_by_level['child']}"

    print("\n[PASS] All global tile IDs are unique")
    print("[PASS] Local tile IDs correctly start from 0 at each level")
    print()

    # Print tree for visual inspection
    print("Tree structure:")
    tree.print_tree()
    print()


def test_deep_recursive_tree():
    """Test a deeper recursive tree (3 levels)."""
    print("=" * 80)
    print("Test 2: Deep Recursive Tree (3 levels)")
    print("=" * 80)

    # Create a 3-level tree
    # Level 0 (root): 4x4 grid (16 tiles), 4 channels
    # - Leaf: 12 tiles (3 per channel)
    # - Internal child 1: 4 tiles (2x2 grid)
    #   - Leaf: 2 tiles
    #   - Internal child 2: 2 tiles (1x2 grid)
    #     - Leaf: 2 tiles

    channel_ids = ["ch0", "ch1", "ch2", "ch3"]

    tree = MatrixAllocationTree.create_root(
        rows=16,
        cols=16,
        batch_size=1,
        num_split_row=4,
        num_split_col=4,
        channel_ids=channel_ids
    )

    # Level 0: 12 tiles to leaf
    tree.root.add_leaf_child(num_tiles=12)

    # Level 0: 4 tiles to internal child
    level1_child = tree.root.add_internal_child(
        rows=8,
        cols=8,
        batch_size=1,
        num_split_row=2,
        num_split_col=2,
        num_source_tiles=4,
        source_grid_shape=(2, 2)
    )

    # Level 1: 2 tiles to leaf
    level1_child.add_leaf_child(num_tiles=2)

    # Level 1: 2 tiles to internal child
    level2_child = level1_child.add_internal_child(
        rows=4,
        cols=8,
        batch_size=1,
        num_split_row=1,
        num_split_col=2,
        num_source_tiles=2,
        source_grid_shape=(1, 2)
    )

    # Level 2: 2 tiles to leaf
    level2_child.add_leaf_child(num_tiles=2)

    # Validate and assign IDs
    assert tree.validate(check_allocations=False), "Tree structure validation failed"

    success = tree.assign_tile_ids()
    assert success, "Tile ID assignment failed"

    assert tree.validate(check_allocations=True), "Tree validation with allocations failed"

    # Collect all global and local IDs by level
    all_global_ids = []
    local_ids_by_level = {0: [], 1: [], 2: []}

    # Level 0 (root)
    if tree.root.leaf_child:
        for alloc in tree.root.leaf_child.allocations:
            all_global_ids.append(alloc.global_tile_id)
            local_ids_by_level[0].append(alloc.tile_id)

    # Level 1
    for child1 in tree.root.internal_children:
        if child1.leaf_child:
            for alloc in child1.leaf_child.allocations:
                all_global_ids.append(alloc.global_tile_id)
                local_ids_by_level[1].append(alloc.tile_id)

        # Level 2
        for child2 in child1.internal_children:
            if child2.leaf_child:
                for alloc in child2.leaf_child.allocations:
                    all_global_ids.append(alloc.global_tile_id)
                    local_ids_by_level[2].append(alloc.tile_id)

    # Test global ID uniqueness
    print(f"\nTotal tiles: {len(all_global_ids)}")
    print(f"Global IDs: {sorted(all_global_ids)}")

    assert len(all_global_ids) == len(set(all_global_ids)), \
        f"Global tile IDs are not unique!"

    assert set(all_global_ids) == set(range(16)), \
        f"Global IDs don't cover expected range 0-15"

    # Test local IDs at each level
    print(f"\nLevel 0 local IDs: {sorted(local_ids_by_level[0])}")
    print(f"Level 1 local IDs: {sorted(local_ids_by_level[1])}")
    print(f"Level 2 local IDs: {sorted(local_ids_by_level[2])}")

    assert set(local_ids_by_level[0]) == set(range(12)), \
        f"Level 0 local IDs incorrect"

    assert set(local_ids_by_level[1]) == set(range(2)), \
        f"Level 1 local IDs incorrect"

    assert set(local_ids_by_level[2]) == set(range(2)), \
        f"Level 2 local IDs incorrect"

    print("\n[PASS] All global tile IDs are unique across 3 levels")
    print("[PASS] Local tile IDs correctly reset at each level")
    print()

    # Print tree
    print("Tree structure:")
    tree.print_tree()
    print()


def test_two_tail_regions():
    """Test a tree with two tail regions (two internal children at root)."""
    print("=" * 80)
    print("Test 3: Two Tail Regions")
    print("=" * 80)

    # Root: 3x4 grid (12 tiles), 4 channels
    # - Leaf: 8 tiles (2 per channel)
    # - Internal child 1: 2 tiles (1x2 grid)
    #   - Leaf: 2 tiles
    # - Internal child 2: 2 tiles (2x1 grid)
    #   - Leaf: 2 tiles

    channel_ids = ["ch0", "ch1", "ch2", "ch3"]

    tree = MatrixAllocationTree.create_root(
        rows=12,
        cols=12,
        batch_size=1,
        num_split_row=3,
        num_split_col=4,
        channel_ids=channel_ids
    )

    # Root: 8 tiles to leaf
    tree.root.add_leaf_child(num_tiles=8)

    # Tail region 1: 2 tiles (1x2 grid)
    tail1 = tree.root.add_internal_child(
        rows=4,
        cols=6,
        batch_size=1,
        num_split_row=1,
        num_split_col=2,
        num_source_tiles=2,
        source_grid_shape=(1, 2)
    )
    tail1.add_leaf_child(num_tiles=2)

    # Tail region 2: 2 tiles (2x1 grid)
    tail2 = tree.root.add_internal_child(
        rows=8,
        cols=3,
        batch_size=1,
        num_split_row=2,
        num_split_col=1,
        num_source_tiles=2,
        source_grid_shape=(2, 1)
    )
    tail2.add_leaf_child(num_tiles=2)

    # Validate and assign IDs
    assert tree.validate(check_allocations=False), "Tree structure validation failed"

    success = tree.assign_tile_ids()
    assert success, "Tile ID assignment failed"

    assert tree.validate(check_allocations=True), "Tree validation with allocations failed"

    # Collect all global IDs
    all_global_ids = []

    # From root leaf
    if tree.root.leaf_child:
        for alloc in tree.root.leaf_child.allocations:
            all_global_ids.append(alloc.global_tile_id)

    # From both tail regions
    for child in tree.root.internal_children:
        if child.leaf_child:
            for alloc in child.leaf_child.allocations:
                all_global_ids.append(alloc.global_tile_id)

    # Test uniqueness
    print(f"\nTotal tiles: {len(all_global_ids)}")
    print(f"Global IDs: {sorted(all_global_ids)}")

    assert len(all_global_ids) == len(set(all_global_ids)), \
        f"Global tile IDs are not unique!"

    assert set(all_global_ids) == set(range(12)), \
        f"Global IDs don't cover expected range 0-11"

    print("\n[PASS] Global tile IDs are unique with two tail regions")
    print()

    # Print tree
    print("Tree structure:")
    tree.print_tree()
    print()


def test_single_tile_edge_case():
    """Test edge case with single tile."""
    print("=" * 80)
    print("Test 4: Single Tile Edge Case")
    print("=" * 80)

    channel_ids = ["ch0"]

    tree = MatrixAllocationTree.create_root(
        rows=4,
        cols=4,
        batch_size=1,
        num_split_row=1,
        num_split_col=1,
        channel_ids=channel_ids
    )

    tree.root.add_leaf_child(num_tiles=1)

    assert tree.validate(check_allocations=False), "Tree structure validation failed"

    success = tree.assign_tile_ids()
    assert success, "Tile ID assignment failed"

    assert tree.validate(check_allocations=True), "Tree validation with allocations failed"

    # Check IDs
    allocations = tree.get_all_allocations()
    all_global_ids = []
    all_local_ids = []

    for channel_allocs in allocations.values():
        for alloc in channel_allocs:
            all_global_ids.append(alloc.global_tile_id)
            all_local_ids.append(alloc.tile_id)

    print(f"\nGlobal IDs: {all_global_ids}")
    print(f"Local IDs: {all_local_ids}")

    assert all_global_ids == [0], "Single tile should have global ID 0"
    assert all_local_ids == [0], "Single tile should have local ID 0"

    print("\n[PASS] Single tile edge case works correctly")
    print()


def test_large_tree():
    """Test a larger tree with many tiles."""
    print("=" * 80)
    print("Test 5: Large Tree (64 tiles)")
    print("=" * 80)

    # 8x8 grid = 64 tiles, 8 channels
    channel_ids = [f"ch{i}" for i in range(8)]

    tree = MatrixAllocationTree.create_root(
        rows=32,
        cols=32,
        batch_size=1,
        num_split_row=8,
        num_split_col=8,
        channel_ids=channel_ids
    )

    # 56 tiles to leaf (7 per channel)
    tree.root.add_leaf_child(num_tiles=56)

    # 8 tiles to internal child
    tail = tree.root.add_internal_child(
        rows=8,
        cols=32,
        batch_size=1,
        num_split_row=2,
        num_split_col=4,
        num_source_tiles=8,
        source_grid_shape=(1, 8)
    )
    tail.add_leaf_child(num_tiles=8)

    # Validate and assign IDs
    assert tree.validate(check_allocations=False), "Tree structure validation failed"

    success = tree.assign_tile_ids()
    assert success, "Tile ID assignment failed"

    assert tree.validate(check_allocations=True), "Tree validation with allocations failed"

    # Collect all global IDs
    all_global_ids = []

    if tree.root.leaf_child:
        for alloc in tree.root.leaf_child.allocations:
            all_global_ids.append(alloc.global_tile_id)

    for child in tree.root.internal_children:
        if child.leaf_child:
            for alloc in child.leaf_child.allocations:
                all_global_ids.append(alloc.global_tile_id)

    # Test uniqueness
    print(f"\nTotal tiles: {len(all_global_ids)}")
    print(f"Unique global IDs: {len(set(all_global_ids))}")

    assert len(all_global_ids) == 64, "Should have 64 tiles"
    assert len(all_global_ids) == len(set(all_global_ids)), \
        f"Global tile IDs are not unique!"

    assert set(all_global_ids) == set(range(64)), \
        f"Global IDs don't cover expected range 0-63"

    print("\n[PASS] Large tree with 64 tiles maintains unique global IDs")
    print()


if __name__ == "__main__":
    test_global_tile_id_uniqueness()
    test_deep_recursive_tree()
    test_two_tail_regions()
    test_single_tile_edge_case()
    test_large_tree()

    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print("- Global tile IDs are unique across all tree levels")
    print("- Local tile IDs correctly start from 0 at each node")
    print("- Multi-level recursive structures work correctly")
    print("- Edge cases (single tile, large trees) handled properly")
    print("- Two tail regions maintain ID uniqueness")
