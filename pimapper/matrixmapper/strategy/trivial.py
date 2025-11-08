from dataclasses import dataclass
from math import ceil, sqrt
from typing import List, Optional, Tuple

from ...core.hwspec import Accelerator
from ...core.matrixspec import Mapping, MatrixShape


@dataclass
class TrivialTilingStrategy:
    """Grid-based tiling that assigns submatrices to PIM channels in round-robin order."""

    def create_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        grid_rows: int,
        grid_cols: int,
        batch_splits: int = 1,
    ) -> Mapping:
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers")
        if batch_splits <= 0:
            raise ValueError("batch_splits must be a positive integer")
        if matrix_shape.rows <= 0 or matrix_shape.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if matrix_shape.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if grid_rows > matrix_shape.rows or grid_cols > matrix_shape.cols:
            raise ValueError("grid dimensions cannot exceed matrix dimensions")
        if batch_splits > matrix_shape.batch_size:
            raise ValueError("batch_splits cannot exceed matrix batch_size")
        if not accelerator.channels:
            raise ValueError("Accelerator must have at least one PIM channel")

        row_bounds = self._split_dimension(matrix_shape.rows, grid_rows)
        col_bounds = self._split_dimension(matrix_shape.cols, grid_cols)
        batch_bounds = self._split_dimension(matrix_shape.batch_size, batch_splits)
        if not row_bounds or not col_bounds or not batch_bounds:
            raise ValueError("Grid configuration produced no tiles")

        channel_ids = sorted(accelerator.channels.keys())
        mapping = Mapping(matrix=matrix_shape, accelerator=accelerator)

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
        return mapping

    def create_balanced_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        max_tile_area: Optional[int] = None,
        batch_splits: int = 1,
    ) -> Mapping:
        if not accelerator.channels:
            raise ValueError("Accelerator must have at least one PIM channel")

        num_channels = len(accelerator.channels)
        matrix_volume = matrix_shape.volume

        if max_tile_area is None:
            target_tiles = max(1, num_channels * batch_splits)
        else:
            target_tiles = max(num_channels * batch_splits, ceil(matrix_volume / max_tile_area))

        grid_rows, grid_cols = self._derive_grid(matrix_shape, target_tiles // batch_splits)
        return self.create_mapping(matrix_shape, accelerator, grid_rows, grid_cols, batch_splits)

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
