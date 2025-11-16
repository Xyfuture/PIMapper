from dataclasses import dataclass
from math import ceil, sqrt
from typing import List, Optional, Tuple

from ...core.hwspec import Accelerator
from ...core.matrixspec import Mapping, MatrixShape
from ...core.utils import MappingResult
from ..evaluator import evaluate
from ..utils import MatrixAllocationTree


@dataclass
class TrivialTilingStrategy:
    """Grid-based tiling that assigns submatrices to PIM channels in round-robin order."""

    def create_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[Mapping, MatrixAllocationTree]:
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers")
        if matrix_shape.rows <= 0 or matrix_shape.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if matrix_shape.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if grid_rows > matrix_shape.rows or grid_cols > matrix_shape.cols:
            raise ValueError("grid dimensions cannot exceed matrix dimensions")
        if not accelerator.channels:
            raise ValueError("Accelerator must have at least one PIM channel")

        row_bounds = self._split_dimension(matrix_shape.rows, grid_rows)
        col_bounds = self._split_dimension(matrix_shape.cols, grid_cols)
        # Do not split batch dimension to avoid weight duplication
        batch_bounds = [(0, matrix_shape.batch_size)]
        if not row_bounds or not col_bounds:
            raise ValueError("Grid configuration produced no tiles")

        channel_ids = sorted(accelerator.channels.keys())
        mapping = Mapping(matrix=matrix_shape, accelerator=accelerator)

        # Calculate total tiles (batch is not split, so only spatial tiles)
        total_tiles = len(row_bounds) * len(col_bounds)

        # Create allocation tree
        tree = MatrixAllocationTree.create_root(
            rows=matrix_shape.rows,
            cols=matrix_shape.cols,
            batch_size=matrix_shape.batch_size,
            num_split_row=grid_rows,
            num_split_col=grid_cols
        )

        # Add leaf child with all tiles (single-level tree)
        tree.root.add_leaf_child(num_tiles=total_tiles)

        tile_index = 0
        for row0, row1 in row_bounds:
            for col0, col1 in col_bounds:
                for batch0, batch1 in batch_bounds:
                    channel_id = channel_ids[tile_index % len(channel_ids)]
                    num_rows = row1 - row0
                    num_cols = col1 - col0
                    num_batches = batch1 - batch0
                    mapping.add_tile(channel_id, num_rows, num_cols, num_batches)
                    tile_index += 1

        mapping.check_all()

        return mapping, tree

    def find_optimal_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        grid_rows: int,
        grid_cols: int,
    ) -> Optional[MappingResult]:
        """Find the mapping and evaluate its performance.

        This method creates the mapping with specified grid dimensions and evaluates it to get the latency.

        Args:
            matrix_shape: Matrix dimensions
            accelerator: Hardware accelerator configuration
            grid_rows: Number of row splits
            grid_cols: Number of column splits

        Returns:
            MappingResult with mapping and latency, or None if mapping fails
        """
        try:
            mapping, tree = self.create_mapping(matrix_shape, accelerator, grid_rows, grid_cols)
            latency = evaluate(accelerator, mapping)

            result = MappingResult(mapping=mapping, latency=latency, allocation_tree=tree)
            return result
        except Exception as e:
            return None

    def create_balanced_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        max_tile_area: Optional[int] = None,
    ) -> Tuple[Mapping, MatrixAllocationTree]:
        if not accelerator.channels:
            raise ValueError("Accelerator must have at least one PIM channel")

        num_channels = len(accelerator.channels)
        matrix_area = matrix_shape.rows * matrix_shape.cols

        if max_tile_area is None:
            target_tiles = max(1, num_channels)
        else:
            target_tiles = max(num_channels, ceil(matrix_area / max_tile_area))

        grid_rows, grid_cols = self._derive_grid(matrix_shape, target_tiles)
        return self.create_mapping(matrix_shape, accelerator, grid_rows, grid_cols)

    def _split_dimension(self, total: int, parts: int) -> List[Tuple[int, int]]:
        if parts > total:
            raise ValueError("grid dimension parts cannot exceed matrix dimension")
        base = total // parts
        extra = total % parts
        bounds: List[Tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            length = base + (1 if idx < extra else 0)
            end = start + length
            bounds.append((start, end))
            start = end
        return bounds

    def _derive_grid(self, matrix_shape: MatrixShape, tiles: int) -> Tuple[int, int]:
        # Aim for a near-square grid without exceeding matrix dimensions.
        side = max(1, ceil(sqrt(tiles)))
        grid_rows = min(matrix_shape.rows, side)
        grid_cols = min(matrix_shape.cols, ceil(tiles / grid_rows))
        if grid_cols == 0:
            grid_cols = 1
        return grid_rows, grid_cols
