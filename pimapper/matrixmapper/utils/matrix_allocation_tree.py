"""Matrix allocation tree for tracking tile positions in recursive grid search.

This module implements a tree structure to track how matrix tiles are allocated
to channels during recursive grid search. The tree enables precise tracking of
which tiles are assigned to which channels and their positions in the original matrix.

Tree Structure:
- InternalNode: Represents a matrix to be split, contains grid split information
- LeafNode: Represents channel tiles that have been assigned to channels

The tree is built in two passes:
1. First pass: Build tree structure without specific tile IDs
2. Second pass: Assign tile IDs in row-major order, respecting shape constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("matrix_allocation_tree")


@dataclass
class TileAllocation:
    """Represents a tile allocation with its ID and channel assignment.

    Attributes:
        tile_id: Local identifier within the current node's grid (0-indexed)
        global_tile_id: Global unique identifier in the root matrix (row-major order)
        channel_id: ID of the channel this tile is assigned to
        tile_index_in_grid: Position in the grid (row_idx, col_idx) for position tracking
    """
    tile_id: int
    global_tile_id: int
    channel_id: str
    tile_index_in_grid: Tuple[int, int]  # (row_idx, col_idx) in the grid


@dataclass
class LeafNode:
    """Leaf node representing channel tiles that have been assigned.

    A leaf node contains tiles that have been distributed to channels using
    round-robin allocation. Each tile has a unique ID and is mapped to a channel.

    Attributes:
        num_tiles: Number of tiles in this leaf node
        allocations: List of tile allocations (populated in second pass)
        parent: Reference to parent internal node
    """
    num_tiles: int
    allocations: List[TileAllocation] = field(default_factory=list)
    parent: Optional[InternalNode] = None

    def add_allocation(self, tile_id: int, global_tile_id: int, channel_id: str, grid_pos: Tuple[int, int]) -> None:
        """Add a tile allocation to this leaf node.

        Args:
            tile_id: Local tile ID within the current node's grid
            global_tile_id: Global tile ID in the root matrix
            channel_id: Channel ID for this allocation
            grid_pos: Grid position (row, col) in the current node
        """
        self.allocations.append(TileAllocation(
            tile_id=tile_id,
            global_tile_id=global_tile_id,
            channel_id=channel_id,
            tile_index_in_grid=grid_pos
        ))

    def get_tiles_by_channel(self) -> Dict[str, List[TileAllocation]]:
        """Group tiles by channel ID."""
        result: Dict[str, List[TileAllocation]] = {}
        for alloc in self.allocations:
            if alloc.channel_id not in result:
                result[alloc.channel_id] = []
            result[alloc.channel_id].append(alloc)
        return result

    def validate(self, check_allocations: bool = False) -> bool:
        """Validate that the leaf node has correct number of allocations.

        Args:
            check_allocations: If True, validate that allocations are assigned.
                              If False, only validate structure (for pre-assignment validation).
        """
        if check_allocations and len(self.allocations) != self.num_tiles:
            logger.error(f"LeafNode validation failed: expected {self.num_tiles} tiles, got {len(self.allocations)}")
            return False
        return True


@dataclass
class InternalNode:
    """Internal node representing a matrix to be split.

    An internal node represents a matrix that will be split into tiles using
    a grid configuration. It can have:
    - One leaf node (all tiles assigned to channels)
    - One leaf node + one internal node (some tiles assigned, remainder recursed)
    - One leaf node + two internal nodes (some tiles assigned, two tail regions)

    Attributes:
        rows: Number of rows in this matrix
        cols: Number of columns in this matrix
        batch_size: Batch size for this matrix
        num_split_row: Number of row splits in the grid
        num_split_col: Number of column splits in the grid
        source_tile_ids: Tile IDs that compose this matrix (empty for root)
        source_grid_shape: Shape (m, n) of how source tiles are arranged (None for root)
        leaf_child: Leaf node containing channel-assigned tiles
        internal_children: List of internal nodes for recursive sub-problems
        parent: Reference to parent internal node
    """
    rows: int
    cols: int
    batch_size: int
    num_split_row: int
    num_split_col: int
    source_tile_ids: List[int] = field(default_factory=list)
    source_grid_shape: Optional[Tuple[int, int]] = None  # (m, n) arrangement
    leaf_child: Optional[LeafNode] = None
    internal_children: List[InternalNode] = field(default_factory=list)
    parent: Optional[InternalNode] = None

    def total_tiles(self) -> int:
        """Calculate total number of tiles in this grid."""
        return self.num_split_row * self.num_split_col

    def add_leaf_child(self, num_tiles: int) -> LeafNode:
        """Add a leaf child node."""
        self.leaf_child = LeafNode(num_tiles=num_tiles, parent=self)
        return self.leaf_child

    def add_internal_child(
        self,
        rows: int,
        cols: int,
        batch_size: int,
        num_split_row: int,
        num_split_col: int,
        num_source_tiles: int,
        source_grid_shape: Tuple[int, int]
    ) -> InternalNode:
        """Add an internal child node for recursive processing."""
        child = InternalNode(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            num_split_row=num_split_row,
            num_split_col=num_split_col,
            source_tile_ids=[],  # Will be filled in second pass
            source_grid_shape=source_grid_shape,
            parent=self
        )
        self.internal_children.append(child)
        return child

    def validate(self, check_allocations: bool = False) -> bool:
        """Validate this internal node and its children recursively.

        Args:
            check_allocations: If True, validate that allocations are assigned.
                              If False, only validate structure (for pre-assignment validation).

        Validation rules:
        - Must have at least one leaf child (cannot have zero leaf nodes)
        - Can have 0, 1, or 2 internal children
        - Total children <= 3
        - All children must be valid
        """
        # Must have a leaf child
        if self.leaf_child is None:
            logger.error(f"InternalNode validation failed: no leaf child at node ({self.rows}x{self.cols})")
            return False

        # Validate leaf child
        if not self.leaf_child.validate(check_allocations):
            return False

        # Can have at most 2 internal children
        if len(self.internal_children) > 2:
            logger.error(f"InternalNode validation failed: {len(self.internal_children)} internal children (max 2)")
            return False

        # Validate all internal children recursively
        for child in self.internal_children:
            if not child.validate(check_allocations):
                return False

        # Validate tile count consistency
        total_tiles = self.total_tiles()
        leaf_tiles = self.leaf_child.num_tiles
        internal_tiles = sum(child.total_tiles() for child in self.internal_children)

        if leaf_tiles + internal_tiles != total_tiles:
            logger.error(
                f"InternalNode validation failed: tile count mismatch at ({self.rows}x{self.cols}). "
                f"Total: {total_tiles}, Leaf: {leaf_tiles}, Internal: {internal_tiles}"
            )
            return False

        return True


@dataclass
class MatrixAllocationTree:
    """Tree structure for tracking matrix tile allocations.

    This tree tracks how a matrix is recursively split and allocated to channels.
    It supports two-pass construction:
    1. First pass: Build tree structure with tile counts
    2. Second pass: Assign specific tile IDs in row-major order

    Attributes:
        root: Root internal node representing the original matrix
        num_channels: Number of available channels
        channel_ids: List of channel IDs for round-robin allocation
    """
    root: InternalNode
    num_channels: int
    channel_ids: List[str]

    @classmethod
    def create_root(
        cls,
        rows: int,
        cols: int,
        batch_size: int,
        num_split_row: int,
        num_split_col: int,
        channel_ids: List[str]
    ) -> MatrixAllocationTree:
        """Create a new tree with a root node."""
        root = InternalNode(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            num_split_row=num_split_row,
            num_split_col=num_split_col,
            source_tile_ids=[],
            source_grid_shape=None,
            parent=None
        )
        return cls(
            root=root,
            num_channels=len(channel_ids),
            channel_ids=sorted(channel_ids)
        )

    def validate(self, check_allocations: bool = False) -> bool:
        """Validate the entire tree structure.

        Args:
            check_allocations: If True, validate that allocations are assigned.
                              If False, only validate structure (for pre-assignment validation).
        """
        return self.root.validate(check_allocations)

    def assign_tile_ids(self) -> bool:
        """Second pass: Assign tile IDs to all nodes in the tree.

        This performs a top-down traversal, assigning tile IDs in row-major order.
        For each internal node:
        1. First assign IDs to internal children (they have shape constraints)
        2. Then assign remaining IDs to leaf node using round-robin

        Returns:
            True if assignment successful, False otherwise
        """
        # Start with all tile IDs for the root
        total_tiles = self.root.total_tiles()
        available_ids = list(range(total_tiles))

        # At root level, local IDs are the same as global IDs
        global_id_mapping = {i: i for i in available_ids}

        return self._assign_tile_ids_recursive(self.root, available_ids, global_id_mapping)

    def _assign_tile_ids_recursive(
        self,
        node: InternalNode,
        available_ids: List[int],
        global_id_mapping: Dict[int, int]
    ) -> bool:
        """Recursively assign tile IDs to a node and its children.

        Args:
            node: Internal node to process
            available_ids: List of available local tile IDs in row-major order
            global_id_mapping: Mapping from local tile IDs to global tile IDs

        Returns:
            True if assignment successful, False otherwise
        """
        total_tiles = node.total_tiles()

        if len(available_ids) != total_tiles:
            logger.error(
                f"Tile ID assignment failed: node needs {total_tiles} tiles, "
                f"but {len(available_ids)} available"
            )
            return False

        # Calculate grid dimensions
        num_rows = node.num_split_row
        num_cols = node.num_split_col

        # Step 1: Assign IDs to internal children first (they have shape constraints)
        remaining_ids = available_ids.copy()

        for child in node.internal_children:
            child_tiles_needed = child.total_tiles()

            # Extract tiles that form the required shape for this child
            child_ids = self._extract_tiles_for_shape(
                remaining_ids,
                num_rows,
                num_cols,
                child.source_grid_shape
            )

            if child_ids is None or len(child_ids) != child_tiles_needed:
                logger.error(
                    f"Failed to extract {child_tiles_needed} tiles for internal child "
                    f"with shape {child.source_grid_shape}"
                )
                return False

            # Store source tile IDs in child (these are the global IDs from parent)
            child.source_tile_ids = [global_id_mapping[local_id] for local_id in child_ids]

            # Remove assigned IDs from remaining pool
            remaining_ids = [tid for tid in remaining_ids if tid not in child_ids]

            # Create local-to-global mapping for child
            # Child uses local IDs 0 to child_tiles_needed-1
            # These map to the global IDs from child_ids
            child_local_ids = list(range(child_tiles_needed))
            child_global_mapping = {
                local_id: global_id_mapping[parent_local_id]
                for local_id, parent_local_id in zip(child_local_ids, child_ids)
            }

            # Recursively assign IDs in the child subtree
            if not self._assign_tile_ids_recursive(child, child_local_ids, child_global_mapping):
                return False

        # Step 2: Assign remaining IDs to leaf node using round-robin
        if node.leaf_child is None:
            logger.error("Internal node missing leaf child during ID assignment")
            return False

        leaf_tiles_needed = node.leaf_child.num_tiles

        if len(remaining_ids) != leaf_tiles_needed:
            logger.error(
                f"Leaf tile count mismatch: expected {leaf_tiles_needed}, "
                f"got {len(remaining_ids)} remaining"
            )
            return False

        # Assign tiles to channels using round-robin
        # Note: remaining_ids contains parent's local IDs, but we need to assign
        # local IDs starting from 0 for the leaf node's perspective
        for i, parent_local_id in enumerate(remaining_ids):
            channel_id = self.channel_ids[i % self.num_channels]

            # Calculate grid position using the parent's local ID
            grid_pos = self._tile_id_to_grid_pos(parent_local_id, num_cols)

            # Get the global ID from the mapping
            global_tile_id = global_id_mapping[parent_local_id]

            # Use index i as the local tile ID (0-indexed within this leaf node)
            local_tile_id = i

            node.leaf_child.add_allocation(local_tile_id, global_tile_id, channel_id, grid_pos)

        return True

    def _extract_tiles_for_shape(
        self,
        available_ids: List[int],
        grid_rows: int,
        grid_cols: int,
        target_shape: Optional[Tuple[int, int]]
    ) -> Optional[List[int]]:
        """Extract tiles from available IDs that form the target shape.

        This function identifies which tile IDs should be used to form a
        sub-matrix with the specified shape. The tiles must form a contiguous
        rectangular region in the grid.

        Args:
            available_ids: Available tile IDs in row-major order
            grid_rows: Number of rows in the parent grid
            grid_cols: Number of columns in the parent grid
            target_shape: (m, n) shape of tiles to extract

        Returns:
            List of tile IDs that form the shape, or None if extraction fails
        """
        if target_shape is None:
            return None

        m, n = target_shape
        needed_tiles = m * n

        if len(available_ids) < needed_tiles:
            return None

        # Convert available IDs to grid positions
        id_to_pos = {tid: self._tile_id_to_grid_pos(tid, grid_cols) for tid in available_ids}

        # Find a contiguous rectangular region of size m x n
        # Try to find the region in the remaining tiles
        # For tail regions, they are typically at the bottom or right edge

        # Strategy: Find the minimum bounding box that contains exactly m*n tiles
        positions = [id_to_pos[tid] for tid in available_ids]

        if len(positions) < needed_tiles:
            return None

        # Sort positions by row-major order
        positions_sorted = sorted(positions, key=lambda p: (p[0], p[1]))

        # Try to find a contiguous m x n region
        # For tail regions, we expect them to be contiguous
        extracted_ids = []

        # Check if we can form an m x n rectangle from the available positions
        # This is a simplified approach - assumes tail regions are contiguous
        for i in range(len(positions_sorted)):
            if len(extracted_ids) >= needed_tiles:
                break

            r, c = positions_sorted[i]

            # Try to form an m x n rectangle starting from this position
            candidate_ids = []
            valid = True

            for dr in range(m):
                for dc in range(n):
                    target_pos = (r + dr, c + dc)
                    # Find tile ID at this position
                    matching_ids = [tid for tid, pos in id_to_pos.items() if pos == target_pos]

                    if len(matching_ids) != 1:
                        valid = False
                        break

                    candidate_ids.append(matching_ids[0])

                if not valid:
                    break

            if valid and len(candidate_ids) == needed_tiles:
                return candidate_ids

        # Fallback: just take the first needed_tiles in row-major order
        return sorted(available_ids[:needed_tiles])

    def _tile_id_to_grid_pos(self, tile_id: int, grid_cols: int) -> Tuple[int, int]:
        """Convert tile ID to grid position (row, col)."""
        row = tile_id // grid_cols
        col = tile_id % grid_cols
        return (row, col)

    def get_all_allocations(self) -> Dict[str, List[TileAllocation]]:
        """Get all tile allocations grouped by channel.

        Returns:
            Dictionary mapping channel IDs to lists of tile allocations
        """
        result: Dict[str, List[TileAllocation]] = {cid: [] for cid in self.channel_ids}
        self._collect_allocations_recursive(self.root, result)
        return result

    def _collect_allocations_recursive(
        self,
        node: InternalNode,
        result: Dict[str, List[TileAllocation]]
    ) -> None:
        """Recursively collect all allocations from the tree."""
        # Collect from leaf child
        if node.leaf_child:
            for alloc in node.leaf_child.allocations:
                result[alloc.channel_id].append(alloc)

        # Recursively collect from internal children
        for child in node.internal_children:
            self._collect_allocations_recursive(child, result)

    def print_tree(self, node: Optional[InternalNode] = None, indent: int = 0) -> None:
        """Print the tree structure for debugging.

        Args:
            node: Node to print (defaults to root)
            indent: Indentation level
        """
        if node is None:
            node = self.root

        prefix = "  " * indent
        print(f"{prefix}InternalNode: {node.rows}x{node.cols}x{node.batch_size}, "
              f"grid={node.num_split_row}x{node.num_split_col}, "
              f"total_tiles={node.total_tiles()}")

        if node.source_tile_ids:
            print(f"{prefix}  source_tiles={node.source_tile_ids}, "
                  f"shape={node.source_grid_shape}")

        if node.leaf_child:
            print(f"{prefix}  LeafNode: {node.leaf_child.num_tiles} tiles")
            if node.leaf_child.allocations:
                tiles_by_channel = node.leaf_child.get_tiles_by_channel()
                for channel_id, allocs in sorted(tiles_by_channel.items()):
                    tile_ids_local = [a.tile_id for a in allocs]
                    tile_ids_global = [a.global_tile_id for a in allocs]
                    print(f"{prefix}    {channel_id}: local={tile_ids_local}, global={tile_ids_global}")

        for i, child in enumerate(node.internal_children):
            print(f"{prefix}  InternalChild[{i}]:")
            self.print_tree(child, indent + 2)


def create_allocation_tree_from_split(
    matrix_rows: int,
    matrix_cols: int,
    batch_size: int,
    num_split_row: int,
    num_split_col: int,
    channel_ids: List[str],
    tiles_per_channel: int,
    num_tail_regions: int,
    tail_region_shapes: List[Tuple[int, int, int, int, Tuple[int, int]]]
) -> MatrixAllocationTree:
    """Helper function to create an allocation tree from split configuration.

    This is a convenience function for the first pass of tree construction.

    Args:
        matrix_rows: Matrix rows
        matrix_cols: Matrix columns
        batch_size: Batch size
        num_split_row: Number of row splits
        num_split_col: Number of column splits
        channel_ids: List of channel IDs
        tiles_per_channel: Number of tiles assigned per channel (round-robin)
        num_tail_regions: Number of tail regions (0, 1, or 2)
        tail_region_shapes: List of (rows, cols, batch, num_tiles, grid_shape) for each tail region

    Returns:
        MatrixAllocationTree with structure built (IDs not yet assigned)
    """
    tree = MatrixAllocationTree.create_root(
        rows=matrix_rows,
        cols=matrix_cols,
        batch_size=batch_size,
        num_split_row=num_split_row,
        num_split_col=num_split_col,
        channel_ids=channel_ids
    )

    # Calculate number of tiles in leaf node
    total_tiles = num_split_row * num_split_col
    num_channels = len(channel_ids)
    leaf_tiles = tiles_per_channel * num_channels

    # Add leaf child
    tree.root.add_leaf_child(leaf_tiles)

    # Add internal children for tail regions
    for rows, cols, batch, num_tiles, grid_shape in tail_region_shapes:
        # Calculate grid splits for this tail region
        # This would need to be provided by the caller
        # For now, use a simple heuristic
        tail_split_row = grid_shape[0]
        tail_split_col = grid_shape[1]

        tree.root.add_internal_child(
            rows=rows,
            cols=cols,
            batch_size=batch,
            num_split_row=tail_split_row,
            num_split_col=tail_split_col,
            num_source_tiles=num_tiles,
            source_grid_shape=grid_shape
        )

    return tree
