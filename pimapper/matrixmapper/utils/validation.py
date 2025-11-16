"""Validation utilities for verifying mapping consistency with allocation tree."""

from typing import Dict, List
import logging

from ...core.matrixspec import Mapping
from .matrix_allocation_tree import MatrixAllocationTree

logger = logging.getLogger("tree_validation")


def validate_mapping_matches_tree(mapping: Mapping, tree: MatrixAllocationTree) -> bool:
    """Validate that a mapping matches its allocation tree structure.

    This function verifies basic consistency between mapping and tree.

    Args:
        mapping: The mapping to validate
        tree: The allocation tree to validate against

    Returns:
        True if mapping is consistent with tree structure
    """
    # Verify that total tiles cover the matrix
    matrix = mapping.matrix
    matrix_area = matrix.rows * matrix.cols * matrix.batch_size

    # Calculate total area covered by mapping tiles
    mapping_area = 0
    for channel_id, tiles in mapping.placement.items():
        for tile in tiles:
            tile_area = tile.num_rows * tile.num_cols * tile.num_batches
            mapping_area += tile_area

    if mapping_area != matrix_area:
        logger.error(
            f"Mapping tiles don't cover matrix: matrix area = {matrix_area}, "
            f"tiles cover {mapping_area}"
        )
        return False

    # Validate tree structure
    if not tree.validate():
        logger.error("Tree structure validation failed")
        return False

    logger.info(f"Validation passed: mapping covers matrix, tree structure is valid")
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

    mapping_tiles = mapping.placement

    print(f"\nMatrix: {mapping.matrix.rows}x{mapping.matrix.cols}x{mapping.matrix.batch_size}")
    print(f"Number of channels in mapping: {len(mapping_tiles)}")

    print("\nPer-channel breakdown:")
    for channel_id in sorted(mapping_tiles.keys()):
        mapping_channel_tiles = mapping_tiles[channel_id]

        print(f"\n  Channel {channel_id}:")
        print(f"    Mapping: {len(mapping_channel_tiles)} tiles")

        if mapping_channel_tiles:
            shapes = [(t.num_rows, t.num_cols, t.num_batches) for t in mapping_channel_tiles]
            print(f"    Tile shapes: {shapes}")

    print("\nTree structure:")
    tree.print_tree()

    print("\n" + "=" * 80)


def get_channel_statistics(mapping: Mapping, tree: MatrixAllocationTree) -> Dict[str, Dict]:
    """Get statistics about channel utilization from mapping.

    Args:
        mapping: The mapping to analyze
        tree: The allocation tree (for compatibility, not used)

    Returns:
        Dictionary mapping channel IDs to statistics dictionaries
    """
    mapping_tiles = mapping.placement
    stats = {}

    for channel_id, tiles in mapping_tiles.items():
        # Calculate total operations for this channel
        total_ops = 0
        for tile in tiles:
            tile_ops = tile.num_rows * tile.num_cols * tile.num_batches
            total_ops += tile_ops

        stats[channel_id] = {
            "num_tiles": len(tiles),
            "total_operations": total_ops,
            "tile_shapes": [(t.num_rows, t.num_cols, t.num_batches) for t in tiles]
        }

    return stats
