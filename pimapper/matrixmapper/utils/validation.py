"""Validation utilities for verifying mapping consistency with allocation tree."""

from typing import Dict, List
import logging

from ...core.matrixspec import Mapping
from .matrix_allocation_tree import MatrixAllocationTree, TileAllocation

logger = logging.getLogger("tree_validation")


def validate_mapping_matches_tree(mapping: Mapping, tree: MatrixAllocationTree) -> bool:
    """Validate that a mapping matches its allocation tree.

    This function verifies that:
    1. Each channel in the mapping has the same tiles as in the tree
    2. The number of tiles per channel matches
    3. The tile shapes match between mapping and tree

    Args:
        mapping: The mapping to validate
        tree: The allocation tree to validate against

    Returns:
        True if mapping matches tree, False otherwise
    """
    # Get allocations from tree
    tree_allocations = tree.get_all_allocations()

    # Get tiles from mapping
    mapping_tiles = mapping.placement

    # Check that all channels in tree are in mapping
    for channel_id in tree_allocations.keys():
        if channel_id not in mapping_tiles:
            logger.error(f"Channel {channel_id} in tree but not in mapping")
            return False

    # Check that all channels in mapping are in tree
    for channel_id in mapping_tiles.keys():
        if channel_id not in tree_allocations:
            logger.error(f"Channel {channel_id} in mapping but not in tree")
            return False

    # Check tile counts per channel
    for channel_id in tree_allocations.keys():
        tree_tile_count = len(tree_allocations[channel_id])
        mapping_tile_count = len(mapping_tiles[channel_id])

        if tree_tile_count != mapping_tile_count:
            logger.error(
                f"Channel {channel_id}: tree has {tree_tile_count} tiles, "
                f"mapping has {mapping_tile_count} tiles"
            )
            return False

    # Verify total tile count matches matrix
    total_tree_tiles = sum(len(allocs) for allocs in tree_allocations.values())
    total_mapping_tiles = sum(len(tiles) for tiles in mapping_tiles.values())

    if total_tree_tiles != total_mapping_tiles:
        logger.error(
            f"Total tile count mismatch: tree has {total_tree_tiles}, "
            f"mapping has {total_mapping_tiles}"
        )
        return False

    # Verify that total tiles cover the matrix
    matrix = mapping.matrix
    matrix_area = matrix.rows * matrix.cols * matrix.batch_size

    # Calculate total area covered by mapping tiles
    mapping_area = 0
    for channel_id, tiles in mapping_tiles.items():
        for tile in tiles:
            tile_area = tile.num_rows * tile.num_cols * tile.num_batches
            mapping_area += tile_area

    if mapping_area != matrix_area:
        logger.error(
            f"Mapping tiles don't cover matrix: matrix area = {matrix_area}, "
            f"tiles cover {mapping_area}"
        )
        return False

    logger.info(f"Validation passed: {total_tree_tiles} tiles across {len(tree_allocations)} channels")
    return True


def print_mapping_tree_comparison(mapping: Mapping, tree: MatrixAllocationTree) -> None:
    """Print a comparison of mapping and tree for debugging.

    Args:
        mapping: The mapping to compare
        tree: The allocation tree to compare
    """
    print("=" * 80)
    print("Mapping vs Tree Comparison")
    print("=" * 80)

    tree_allocations = tree.get_all_allocations()
    mapping_tiles = mapping.placement

    print(f"\nMatrix: {mapping.matrix.rows}x{mapping.matrix.cols}x{mapping.matrix.batch_size}")
    print(f"Number of channels: {len(tree_allocations)}")

    print("\nPer-channel breakdown:")
    for channel_id in sorted(tree_allocations.keys()):
        tree_allocs = tree_allocations[channel_id]
        mapping_channel_tiles = mapping_tiles.get(channel_id, [])

        print(f"\n  Channel {channel_id}:")
        print(f"    Tree: {len(tree_allocs)} tiles")
        print(f"    Mapping: {len(mapping_channel_tiles)} tiles")

        if tree_allocs:
            tile_ids_local = [a.tile_id for a in tree_allocs]
            tile_ids_global = [a.global_tile_id for a in tree_allocs]
            positions = [a.tile_index_in_grid for a in tree_allocs]
            print(f"    Tree tile IDs (local): {tile_ids_local}")
            print(f"    Tree tile IDs (global): {tile_ids_global}")
            print(f"    Tree positions: {positions}")

        if mapping_channel_tiles:
            shapes = [(t.num_rows, t.num_cols, t.num_batches) for t in mapping_channel_tiles]
            print(f"    Mapping tile shapes: {shapes}")

    print("\n" + "=" * 80)


def get_channel_statistics(mapping: Mapping, tree: MatrixAllocationTree) -> Dict[str, Dict]:
    """Get statistics about channel utilization from mapping and tree.

    Args:
        mapping: The mapping to analyze
        tree: The allocation tree to analyze

    Returns:
        Dictionary mapping channel IDs to statistics dictionaries
    """
    tree_allocations = tree.get_all_allocations()
    mapping_tiles = mapping.placement

    stats = {}

    for channel_id in tree_allocations.keys():
        tree_allocs = tree_allocations[channel_id]
        mapping_channel_tiles = mapping_tiles.get(channel_id, [])

        # Calculate total operations for this channel
        total_ops = 0
        for tile in mapping_channel_tiles:
            tile_ops = tile.num_rows * tile.num_cols * tile.num_batches
            total_ops += tile_ops

        stats[channel_id] = {
            "num_tiles": len(tree_allocs),
            "tile_ids_local": [a.tile_id for a in tree_allocs],
            "tile_ids_global": [a.global_tile_id for a in tree_allocs],
            "total_operations": total_ops,
            "tile_shapes": [(t.num_rows, t.num_cols, t.num_batches) for t in mapping_channel_tiles]
        }

    return stats
