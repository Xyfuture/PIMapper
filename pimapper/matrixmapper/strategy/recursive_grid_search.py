from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ...core.hwspec import Accelerator
from ...core.matrixspec import Mapping, MatrixShape
from ..evaluator import evaluate
from ...core.utils import MappingResult

logger = logging.getLogger("recursive_grid_search")


@dataclass
class RecursiveGridSearchStrategy:
    """Advanced recursive DSE strategy to find optimal Mapping via grid splits.

    This strategy combines the best features from both GPT and Codex implementations:

    Key improvements and fixes:
    - Geometric precision: Correctly handles tail regions as contiguous rectangles
    - Progress guarantee: Prevents infinite recursion when dies >= tiles
    - Complete round-robin: Only performs full rounds, handles remainder geometrically
    - Robust memoization: Caches results by (matrix_shape, available_dies)
    - Comprehensive validation: Full error handling and mapping verification
    - Exact coordinate mapping: Proper offset handling for sub-matrices
    - Iteration limit: Limits recursion depth with round-robin fallback

    Algorithm flow:
    1. Try all split configurations (num_split_row × num_split_col)
    2. Create tiles in row-major order from grid splits
    3. Distribute tiles using round-robin (complete rounds only)
    4. Handle remaining tiles by identifying exact geometric tail regions
    5. Recursively solve tail regions and combine with main mapping (up to max_iterations)
    6. If max iterations reached, fallback to round-robin distribution
    7. Return the configuration with minimum latency
    """

    num_split_row_candidates: List[int] = field(default_factory=lambda: list(range(1, 9)))
    num_split_col_candidates: List[int] = field(default_factory=lambda: list(range(1, 9)))
    memo: Dict[Tuple, MappingResult] = field(default_factory=dict)
    max_iterations: int = field(default=2)
    enable_fallback_splits: bool = field(default=True)

    # ---------------
    # Helper methods
    # ---------------
    def _log(self, iteration: int, message: str) -> None:
        """Log a message with proper indentation for recursion depth."""
        indent = "  " * iteration
        logger.debug(f"{indent}{message}")

    def _validate_and_evaluate(self, accelerator: Accelerator, mapping: Mapping) -> Optional[MappingResult]:
        """Validate mapping and run evaluation, return None on failure."""
        try:
            mapping.check_all()
            latency = evaluate(accelerator, mapping)
            return MappingResult(mapping=mapping, latency=latency)
        except Exception as e:
            logger.error(f"Validation/evaluation failed: {type(e).__name__}: {e}")
            logger.debug(f"Failed mapping details: {mapping}")
            return None

    def _is_valid_split(self, num_r: int, num_c: int, matrix: MatrixShape) -> bool:
        """Check if split configuration is valid."""
        return (num_r > 0 and num_c > 0 and
                num_r <= matrix.rows and num_c <= matrix.cols)

    # ---------------
    # Public API
    # ---------------
    def find_optimal_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        available_channels: Optional[Set[str]] = None,
        current_iteration: int = 0,
    ) -> Optional[MappingResult]:
        """Find the optimal mapping for the given matrix and accelerator configuration.

        Args:
            matrix_shape: The matrix dimensions to optimize for
            accelerator: Hardware accelerator configuration with PIM channels
            available_channels: Set of channel IDs to use (defaults to all channels)
            current_iteration: Current recursion depth (internal parameter)

        Returns:
            MappingResult with optimal mapping and latency, or None if no valid mapping found
        """
        if available_channels is None:
            available_channels = set(accelerator.channels.keys())
        if not available_channels:
            return None

        # Log recursion progress
        self._log(current_iteration, f"[Iteration {current_iteration}] Processing matrix: {matrix_shape.rows}x{matrix_shape.cols}x{matrix_shape.batch_size}")
        self._log(current_iteration, f"  Available channels: {sorted(list(available_channels))} (count: {len(available_channels)})")

        # Check memoization cache
        key = self._memo_key(matrix_shape, available_channels)
        if key in self.memo:
            self._log(current_iteration, "  Result retrieved from cache")
            return self.memo[key]

        best: Optional[MappingResult] = None
        best_latency = float("inf")

        # Generate split candidates with fallback options
        split_candidates = []
        for num_r in self.num_split_row_candidates:
            for num_c in self.num_split_col_candidates:
                if self._is_valid_split(num_r, num_c, matrix_shape):
                    split_candidates.append((num_r, num_c))

        # # Add fallback splits: 1×n_channels and n_channels×1 for perfect channel matching
        # if self.enable_fallback_splits:
        #     n_channels = len(available_channels)
        #     # 1×n_channels: one row, n_channels columns
        #     if n_channels <= matrix_shape.cols and (1, n_channels) not in split_candidates:
        #         self._log(current_iteration, f"  Adding fallback split: 1×{n_channels} (matches channel count)")
        #         split_candidates.append((1, n_channels))
        #     # n_channels×1: n_channels rows, one column
        #     if n_channels <= matrix_shape.rows and (n_channels, 1) not in split_candidates:
        #         self._log(current_iteration, f"  Adding fallback split: {n_channels}×1 (matches channel count)")
        #         split_candidates.append((n_channels, 1))

        # Try all split configurations
        split_count = 0
        for num_r, num_c in split_candidates:
            # Skip invalid configurations
            if not self._is_valid_split(num_r, num_c, matrix_shape):
                continue

            split_count += 1
            self._log(current_iteration, f"  Trying split config {split_count}: {num_r}x{num_c} (rows x cols)")

            # Pruning: Check utilization upper bound
            if best:
                estimated_util = self._estimate_utilization(accelerator, matrix_shape, num_r, num_c)
                current_best_util = best.get_compute_utilization()
                # print(f"  Estimated utilization: {estimated_util:.2%} Current best: {current_best_util:.2%}"   )
                if estimated_util < current_best_util:
                    self._log(current_iteration, f"    Pruned: estimated util {estimated_util:.4f} < best util {current_best_util:.4f}")
                    print(f"    Pruned: estimated util {estimated_util:.4f} < best util {current_best_util:.4f}"   )
                    continue

            try:
                candidate = self._evaluate_split_configuration(
                    matrix_shape, accelerator, available_channels, num_r, num_c, current_iteration
                )
            except Exception:
                self._log(current_iteration, "    Configuration failed")
                candidate = None

            if candidate and candidate.latency < best_latency:
                self._log(current_iteration, f"    Found better config, latency: {candidate.latency}")
                best = candidate
                best_latency = candidate.latency
            elif candidate:
                self._log(current_iteration, f"    Valid config, latency: {candidate.latency}")
            else:
                logger.debug(f"{'  ' * current_iteration}    Invalid config")

        # Cache the best result
        if best:
            self.memo[key] = best
            self._log(current_iteration, f"[Iteration {current_iteration}] Complete, best latency: {best.latency}")
        else:
            self._log(current_iteration, f"[Iteration {current_iteration}] Complete, no valid config found")

        return best

    # ---------------
    # Core evaluation logic
    # ---------------
    def _evaluate_split_configuration(
        self,
        matrix: MatrixShape,
        accelerator: Accelerator,
        available_channels: Set[str],
        num_split_row: int,
        num_split_col: int,
        current_iteration: int,
    ) -> Optional[MappingResult]:
        """Evaluate a specific split configuration and return the mapping result."""

        # Compute split boundaries with ceiling on edge tiles
        row_bounds = self._calculate_split_boundaries(matrix.rows, num_split_row)
        col_bounds = self._calculate_split_boundaries(matrix.cols, num_split_col)
        batch_bounds = [(0, matrix.batch_size)]

        # Create tiles in row-major order (critical for correct tail geometry)
        tiles: List[MatrixShape] = []
        for r0, r1 in row_bounds:
            for c0, c1 in col_bounds:
                for b0, b1 in batch_bounds:
                    tiles.append(MatrixShape(
                        rows=r1 - r0,
                        cols=c1 - c0,
                        batch_size=b1 - b0
                    ))

        channel_ids = sorted(list(available_channels))
        n_tiles = len(tiles)
        n_channels = len(channel_ids)

        self._log(current_iteration, f"    Generated {n_tiles} tiles, {n_channels} channels")

        if n_channels == 0:
            return None

        # Progress guarantee: if we have enough channels, assign all tiles
        if n_tiles <= n_channels:
            self._log(current_iteration, f"    Tiles <= channels, direct assignment")
            assignments: Dict[str, List[MatrixShape]] = {c: [] for c in channel_ids}
            for i, tile in enumerate(tiles):
                assignments[channel_ids[i % n_channels]].append(tile)

            main_mapping = self._create_mapping_from_specs(matrix, accelerator, assignments)
            return self._validate_and_evaluate(accelerator, main_mapping)

        # Round-robin distribution: complete rounds only
        tiles_per_channel = n_tiles // n_channels
        assigned_count = tiles_per_channel * n_channels

        self._log(current_iteration, f"    Round-robin: {tiles_per_channel} tiles per channel, total {assigned_count} assigned")

        main_assignments: Dict[str, List[MatrixShape]] = {c: [] for c in channel_ids}
        for i in range(assigned_count):
            main_assignments[channel_ids[i % n_channels]].append(tiles[i])

        remaining_tiles = tiles[assigned_count:]
        main_mapping = self._create_mapping_from_specs(matrix, accelerator, main_assignments)

        # If no remaining tiles, we're done
        if not remaining_tiles:
            self._log(current_iteration, f"    No remaining tiles, assignment complete")
            return self._validate_and_evaluate(accelerator, main_mapping)

        self._log(current_iteration, f"    Remaining {len(remaining_tiles)} tiles for recursive processing")

        # Check iteration limit before recursion
        if current_iteration >= self.max_iterations:
            self._log(current_iteration, f"    Max recursion depth ({self.max_iterations}) reached, using round-robin fallback")
            # Fallback: Use round-robin distribution for remaining tiles
            return self._round_robin_fallback(matrix, accelerator, main_assignments, remaining_tiles)

        # Handle tail tiles by constructing exact geometric sub-regions
        sub_shapes = self._construct_tail_subregions(row_bounds, col_bounds, len(remaining_tiles), matrix.batch_size)

        self._log(current_iteration, f"    Constructed {len(sub_shapes)} tail sub-regions")

        # Recursively solve each subregion
        sub_mappings: List[Mapping] = []
        for i, sub_shape in enumerate(sub_shapes):
            self._log(current_iteration, f"    Processing sub-region {i+1}: {sub_shape.rows}x{sub_shape.cols}x{sub_shape.batch_size}")
            sub_result = self.find_optimal_mapping(sub_shape, accelerator, available_channels, current_iteration + 1)
            if not sub_result:
                self._log(current_iteration, f"    Sub-region {i+1} has no solution")
                return None
            # Simply copy the mapping (no coordinate offset needed since Tile has no position)
            sub_mappings.append(self._copy_mapping(sub_result.mapping))

        # Combine main mapping with sub-mappings
        combined_mapping = self._combine_mappings(main_mapping, sub_mappings)
        return self._validate_and_evaluate(accelerator, combined_mapping)

    # ---------------
    # Geometry and tiling helpers
    # ---------------
    def _calculate_split_boundaries(self, dimension: int, num_splits: int) -> List[Tuple[int, int]]:
        """Calculate split boundaries with ceiling operation for edge tiles."""
        if num_splits <= 0:
            return [(0, dimension)]

        base_size = dimension // num_splits
        remainder = dimension % num_splits

        boundaries = []
        start = 0

        for i in range(num_splits):
            # Distribute remainder to first `remainder` splits (ceiling operation)
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            boundaries.append((start, end))
            start = end

        return boundaries

    def _create_mapping_from_specs(
        self,
        matrix: MatrixShape,
        accelerator: Accelerator,
        assignments: Dict[str, List[MatrixShape]],
    ) -> Mapping:
        """Create a mapping object from MatrixShape assignments."""
        mapping = Mapping(matrix=matrix, accelerator=accelerator)
        for channel_id, shape_list in assignments.items():
            for shape in shape_list:
                mapping.add_tile(channel_id, shape.rows, shape.cols, shape.batch_size)
        return mapping

    def _construct_tail_subregions(
        self,
        row_bounds: List[Tuple[int, int]],
        col_bounds: List[Tuple[int, int]],
        num_remaining_tiles: int,
        batch_size: int,
    ) -> List[MatrixShape]:
        """Construct at most two rectangular subregions that exactly cover the tail.

        The tail in row-major order always consists of:
        - Zero or more complete bottom rows, and
        - An optional right-edge suffix of the row just above them.

        This geometric precision is critical for correctness.
        """
        if num_remaining_tiles == 0:
            return []

        # Grid dimensions
        g_rows = len(row_bounds)
        g_cols = len(col_bounds)

        # Analyze tail structure
        full_rows = num_remaining_tiles // g_cols
        suffix_cols = num_remaining_tiles % g_cols

        # Define cases for data-driven approach
        cases = [
            # Case 1: Only partial row suffix
            (full_rows == 0 and suffix_cols > 0,
             [(g_rows - 1, g_rows, g_cols - suffix_cols, g_cols)]),

            # Case 2: Only complete bottom rows
            (suffix_cols == 0 and full_rows > 0,
             [(g_rows - full_rows, g_rows, 0, g_cols)]),

            # Case 3: Partial row + complete rows below
            (suffix_cols > 0 and full_rows > 0, [
                (g_rows - full_rows - 1, g_rows - full_rows, g_cols - suffix_cols, g_cols),
                (g_rows - full_rows, g_rows, 0, g_cols)
            ])
        ]

        # Process the matching case
        for condition, regions in cases:
            if condition:
                shapes = []
                for r in regions:
                    shape = self._grid_rect_to_shape(row_bounds, col_bounds, *r, batch_size)
                    if shape is not None:
                        shapes.append(shape)
                return shapes

        return []

    def _grid_rect_to_shape(
        self,
        row_bounds: List[Tuple[int, int]],
        col_bounds: List[Tuple[int, int]],
        r0_idx: int,
        r1_idx_excl: int,
        c0_idx: int,
        c1_idx_excl: int,
        batch_size: int = 1
    ) -> Optional[MatrixShape]:
        """Convert grid indices to matrix shape."""
        if r0_idx >= r1_idx_excl or c0_idx >= c1_idx_excl:
            return None
        row0 = row_bounds[r0_idx][0]
        row1 = row_bounds[r1_idx_excl - 1][1]
        col0 = col_bounds[c0_idx][0]
        col1 = col_bounds[c1_idx_excl - 1][1]
        return MatrixShape(rows=row1 - row0, cols=col1 - col0, batch_size=batch_size)

    def _copy_mapping(self, mapping: Mapping) -> Mapping:
        """Copy a mapping (since Tile has no position information, this is straightforward)."""
        copied_mapping = Mapping(matrix=mapping.matrix, chip=mapping.chip)
        for die_id, tiles in mapping.placement.items():
            for tile in tiles:
                copied_mapping.add_tile(
                    die_id,
                    tile.num_rows,
                    tile.num_cols,
                    tile.num_batches,
                )
        return copied_mapping

    def _combine_mappings(self, main_mapping: Mapping, sub_mappings: List[Mapping]) -> Mapping:
        """Combine main mapping with sub-region mappings."""
        combined = Mapping(matrix=main_mapping.matrix, chip=main_mapping.chip)

        # Add tiles from main mapping
        for die_id, tiles in main_mapping.placement.items():
            for tile in tiles:
                combined.add_tile(
                    die_id, tile.num_rows, tile.num_cols, tile.num_batches
                )

        # Add tiles from sub-mappings
        for sub_mapping in sub_mappings:
            for die_id, tiles in sub_mapping.placement.items():
                for tile in tiles:
                    combined.add_tile(
                        die_id, tile.num_rows, tile.num_cols, tile.num_batches
                    )

        return combined

    # ---------------
    # Memoization
    # ---------------
    def _memo_key(self, matrix: MatrixShape, available_channels: Set[str]) -> Tuple:
        """Create a memoization key for the subproblem."""
        return (matrix.rows, matrix.cols, matrix.batch_size, tuple(sorted(available_channels)))

    def _round_robin_fallback(
        self,
        matrix: MatrixShape,
        accelerator: Accelerator,
        main_assignments: Dict[str, List[MatrixShape]],
        remaining_tiles: List[MatrixShape]
    ) -> Optional[MappingResult]:
        """Fallback strategy: distribute remaining tiles using round-robin."""
        channel_ids = sorted(list(main_assignments.keys()))

        # Add remaining tiles using round-robin
        for i, tile_shape in enumerate(remaining_tiles):
            channel_id = channel_ids[i % len(channel_ids)]
            main_assignments[channel_id].append(tile_shape)

        # Create and validate the mapping
        combined_mapping = self._create_mapping_from_specs(matrix, accelerator, main_assignments)
        return self._validate_and_evaluate(accelerator, combined_mapping)
    
    def _estimate_utilization(
            self,
            accelerator: Accelerator,
            matrix: MatrixShape,
            num_row_splits: int,
            num_col_splits: int
        ) -> float:

        channel_spec = accelerator.spec.channel_spec
        tile_rows = matrix.rows // num_row_splits + (1 if matrix.rows % num_row_splits != 0 else 0)
        tile_cols = matrix.cols // num_col_splits + (1 if matrix.cols % num_col_splits != 0 else 0)
        batch_size = matrix.batch_size

        input_optimal_latency = (batch_size * tile_rows * matrix.data_format.input_dtype.bytes_per_element) / (channel_spec.get_input_bandwidth() / 2)
        compute_optimal_latency = max(
            tile_rows * tile_cols * matrix.data_format.weight_dtype.bytes_per_element / channel_spec.memory_bandwidth,
            tile_rows * tile_cols * batch_size / channel_spec.compute_power
        ) * 1000
        output_optimal_latency = (batch_size * tile_cols * matrix.data_format.output_dtype.bytes_per_element) / (channel_spec.get_output_bandwidth() / 2)

        optimal_latency = max(input_optimal_latency, output_optimal_latency)

        utilization = (batch_size * tile_rows * tile_cols) / (optimal_latency * channel_spec.compute_power / 1000)

        return utilization