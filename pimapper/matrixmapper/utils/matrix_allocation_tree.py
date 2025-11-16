"""Matrix allocation tree for tracking tile allocation structure in recursive grid search.

This module implements a tree structure to track how matrix tiles are allocated
to channels during recursive grid search.

Tree Structure:
- InternalNode: Represents a matrix to be split, contains grid split information
- LeafNode: Represents tiles that have been assigned to channels (non-tail tiles)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("matrix_allocation_tree")


@dataclass
class LeafNode:
    """Leaf node representing tiles assigned to channels.

    Attributes:
        num_tiles: Number of tiles in this leaf node
    """
    num_tiles: int

    def validate(self) -> bool:
        """Validate that the leaf node has valid tile count."""
        if self.num_tiles <= 0:
            logger.error(f"LeafNode validation failed: num_tiles must be positive, got {self.num_tiles}")
            return False
        return True


@dataclass
class InternalNode:
    """Internal node representing a matrix to be split.

    An internal node represents a matrix that will be split into tiles using
    a grid configuration. It can have:
    - One leaf node (all tiles assigned to channels)
    - One leaf node + internal children (some tiles assigned, remainder recursed)

    Attributes:
        rows: Number of rows in this matrix
        cols: Number of columns in this matrix
        batch_size: Batch size for this matrix
        num_split_row: Number of row splits in the grid
        num_split_col: Number of column splits in the grid
        num_parent_tiles: Number of tiles this node occupies in parent's grid (None for root)
        leaf_child: Leaf node containing channel-assigned tiles
        internal_children: List of internal nodes for recursive sub-problems
    """
    rows: int
    cols: int
    batch_size: int
    num_split_row: int
    num_split_col: int
    num_parent_tiles: Optional[int] = None  # How many parent grid tiles this node covers
    leaf_child: Optional[LeafNode] = None
    internal_children: List[InternalNode] = field(default_factory=list)

    def total_tiles(self) -> int:
        """Calculate total number of tiles in this grid."""
        return self.num_split_row * self.num_split_col

    def add_leaf_child(self, num_tiles: int) -> LeafNode:
        """Add a leaf child node."""
        self.leaf_child = LeafNode(num_tiles=num_tiles)
        return self.leaf_child

    def add_internal_child(
        self,
        rows: int,
        cols: int,
        batch_size: int,
        num_split_row: int,
        num_split_col: int,
        num_parent_tiles: int
    ) -> InternalNode:
        """Add an internal child node for recursive processing.

        Args:
            rows: Matrix rows for this child
            cols: Matrix cols for this child
            batch_size: Batch size for this child
            num_split_row: Number of row splits in child's grid
            num_split_col: Number of column splits in child's grid
            num_parent_tiles: Number of tiles this child occupies in parent's grid
        """
        child = InternalNode(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            num_split_row=num_split_row,
            num_split_col=num_split_col,
            num_parent_tiles=num_parent_tiles
        )
        self.internal_children.append(child)
        return child

    def validate(self) -> bool:
        """Validate this internal node and its children recursively.

        Validation rules:
        - Must have at least one leaf child
        - Can have 0, 1, or 2 internal children
        - All children must be valid
        - Tile count must be consistent (leaf tiles + sum of children's num_parent_tiles = total tiles)
        """
        # Must have a leaf child
        if self.leaf_child is None:
            logger.error(f"InternalNode validation failed: no leaf child at node ({self.rows}x{self.cols})")
            return False

        # Validate leaf child
        if not self.leaf_child.validate():
            return False

        # Can have at most 2 internal children
        if len(self.internal_children) > 2:
            logger.error(f"InternalNode validation failed: {len(self.internal_children)} internal children (max 2)")
            return False

        # Validate all internal children recursively
        for child in self.internal_children:
            if not child.validate():
                return False

        # Validate tile count consistency
        # For root node, num_parent_tiles is None, so we check total_tiles()
        # For child nodes, we check that leaf tiles + children's parent tiles = total tiles
        total_tiles = self.total_tiles()
        leaf_tiles = self.leaf_child.num_tiles
        internal_parent_tiles = sum(child.num_parent_tiles for child in self.internal_children)

        if leaf_tiles + internal_parent_tiles != total_tiles:
            logger.error(
                f"InternalNode validation failed: tile count mismatch at ({self.rows}x{self.cols}). "
                f"Total: {total_tiles}, Leaf: {leaf_tiles}, Internal (parent tiles): {internal_parent_tiles}"
            )
            return False

        return True


@dataclass
class MatrixAllocationTree:
    """Tree structure for tracking matrix tile allocations.

    This tree tracks how a matrix is recursively split and allocated to channels.

    Attributes:
        root: Root internal node representing the original matrix
    """
    root: InternalNode

    @classmethod
    def create_root(
        cls,
        rows: int,
        cols: int,
        batch_size: int,
        num_split_row: int,
        num_split_col: int
    ) -> MatrixAllocationTree:
        """Create a new tree with a root node."""
        root = InternalNode(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            num_split_row=num_split_row,
            num_split_col=num_split_col
        )
        return cls(root=root)

    def validate(self) -> bool:
        """Validate the entire tree structure."""
        return self.root.validate()

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

        if node.leaf_child:
            print(f"{prefix}  LeafNode: {node.leaf_child.num_tiles} tiles")

        for i, child in enumerate(node.internal_children):
            print(f"{prefix}  InternalChild[{i}]:")
            self.print_tree(child, indent + 2)
