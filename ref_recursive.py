from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.description import Chip, Mapping, MatrixShape
from ..core.sim_engine import simulate
from ..core.utils import MappingResult

logger = logging.getLogger("agent_grid_search")


@dataclass(frozen=True)
class SubMatrixSpec:
    """A contiguous sub-matrix region within the original matrix.

    Defined by element-space offsets and size. Used to recursively solve the
    remaining (tail) region after full round-robin assignment.
    """

    row0: int
    row1: int
    col0: int
    col1: int
    batch0: int = 0
    batch1: int = 1

    def to_matrix_shape(self) -> MatrixShape:
        return MatrixShape(
            rows=self.row1 - self.row0,
            cols=self.col1 - self.col0,
            batch_size=self.batch1 - self.batch0
        )

    @classmethod
    def from_tile_tuple(cls, tile: Tuple[int, int, int, int, int, int]) -> "SubMatrixSpec":
        """Create SubMatrixSpec from a 6-element tile tuple."""
        return cls(*tile)

    @classmethod
    def from_bounds(
        cls,
        row_bounds: Tuple[int, int],
        col_bounds: Tuple[int, int],
        batch_bounds: Tuple[int, int] = (0, 1)
    ) -> "SubMatrixSpec":
        """Create SubMatrixSpec from boundary tuples."""
        return cls(
            row_bounds[0], row_bounds[1],
            col_bounds[0], col_bounds[1],
            batch_bounds[0], batch_bounds[1]
        )


@dataclass
class AgentGridSearchStrategy:
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

    def _validate_and_simulate(self, chip: Chip, mapping: Mapping) -> Optional[MappingResult]:
        """Validate mapping and run simulation, return None on failure."""
        try:
            mapping.check_all()
            latency = simulate(chip, mapping)
            return MappingResult(mapping=mapping, latency=latency)
        except Exception as e:
            logger.error(f"Validation/simulation failed: {type(e).__name__}: {e}")
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
        chip: Chip,
        available_dies: Optional[Set[str]] = None,
        current_iteration: int = 0,
    ) -> Optional[MappingResult]:
        """Find the optimal mapping for the given matrix and chip configuration.

        Args:
            matrix_shape: The matrix dimensions to optimize for
            chip: Hardware chip configuration with compute dies
            available_dies: Set of die IDs to use (defaults to all dies)
            current_iteration: Current recursion depth (internal parameter)

        Returns:ai

            MappingResult with optimal mapping and latency, or None if no valid mapping found
        """
        if available_dies is None:
            available_dies = set(chip.compute_dies.keys())
        if not available_dies:
            return None

        # Log recursion progress
        self._log(current_iteration, f"[Iteration {current_iteration}] Processing matrix: {matrix_shape.rows}x{matrix_shape.cols}x{matrix_shape.batch_size}")
        self._log(current_iteration, f"  Available dies: {sorted(list(available_dies))} (count: {len(available_dies)})")

        # Check memoization cache
        key = self._memo_key(matrix_shape, available_dies)
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

        # Add fallback splits: 1×n_dies and n_dies×1 for perfect die matching
        if self.enable_fallback_splits:
            n_dies = len(available_dies)
            # 1×n_dies: one row, n_dies columns
            if n_dies <= matrix_shape.cols and (1, n_dies) not in split_candidates:
                self._log(current_iteration, f"  Adding fallback split: 1×{n_dies} (matches die count)")
                split_candidates.append((1, n_dies))
            # n_dies×1: n_dies rows, one column
            if n_dies <= matrix_shape.rows and (n_dies, 1) not in split_candidates:
                self._log(current_iteration, f"  Adding fallback split: {n_dies}×1 (matches die count)")
                split_candidates.append((n_dies, 1))

        # Try all split configurations
        split_count = 0
        for num_r, num_c in split_candidates:
            # Skip invalid configurations
            if not self._is_valid_split(num_r, num_c, matrix_shape):
                continue

            split_count += 1
            self._log(current_iteration, f"  Trying split config {split_count}: {num_r}x{num_c} (rows x cols)")

            try:
                candidate = self._evaluate_split_configuration(
                    matrix_shape, chip, available_dies, num_r, num_c, current_iteration
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
        chip: Chip,
        available_dies: Set[str],
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
        tiles: List[SubMatrixSpec] = []
        for r0, r1 in row_bounds:
            for c0, c1 in col_bounds:
                for b0, b1 in batch_bounds:
                    tiles.append(SubMatrixSpec(r0, r1, c0, c1, b0, b1))

        die_ids = sorted(list(available_dies))
        n_tiles = len(tiles)
        n_dies = len(die_ids)

        self._log(current_iteration, f"    Generated {n_tiles} tiles, {n_dies} dies")

        if n_dies == 0:
            return None

        # Progress guarantee: if we have enough dies, assign all tiles
        if n_tiles <= n_dies:
            self._log(current_iteration, f"    Tiles <= dies, direct assignment")
            assignments: Dict[str, List[SubMatrixSpec]] = {d: [] for d in die_ids}
            for i, tile in enumerate(tiles):
                assignments[die_ids[i % n_dies]].append(tile)

            main_mapping = self._create_mapping_from_specs(matrix, chip, assignments)
            return self._validate_and_simulate(chip, main_mapping)

        # Round-robin distribution: complete rounds only
        tiles_per_die = n_tiles // n_dies
        assigned_count = tiles_per_die * n_dies

        self._log(current_iteration, f"    Round-robin: {tiles_per_die} tiles per die, total {assigned_count} assigned")

        main_assignments: Dict[str, List[SubMatrixSpec]] = {d: [] for d in die_ids}
        for i in range(assigned_count):
            main_assignments[die_ids[i % n_dies]].append(tiles[i])

        remaining_tiles = tiles[assigned_count:]
        main_mapping = self._create_mapping_from_specs(matrix, chip, main_assignments)

        # If no remaining tiles, we're done
        if not remaining_tiles:
            self._log(current_iteration, f"    No remaining tiles, assignment complete")
            return self._validate_and_simulate(chip, main_mapping)

        self._log(current_iteration, f"    Remaining {len(remaining_tiles)} tiles for recursive processing")

        # Check iteration limit before recursion
        if current_iteration >= self.max_iterations:
            self._log(current_iteration, f"    Max recursion depth ({self.max_iterations}) reached, using round-robin fallback")
            # Fallback: Use round-robin distribution for remaining tiles
            return self._round_robin_fallback(matrix, chip, main_assignments, remaining_tiles)

        # Handle tail tiles by constructing exact geometric sub-regions
        sub_specs = self._construct_tail_subregions(row_bounds, col_bounds, remaining_tiles)

        self._log(current_iteration, f"    Constructed {len(sub_specs)} tail sub-regions")

        # Recursively solve each subregion
        sub_mappings: List[Mapping] = []
        for i, spec in enumerate(sub_specs):
            self._log(current_iteration, f"    Processing sub-region {i+1}: [{spec.row0}:{spec.row1}, {spec.col0}:{spec.col1}, {spec.batch0}:{spec.batch1}]")
            sub_shape = spec.to_matrix_shape()
            sub_result = self.find_optimal_mapping(sub_shape, chip, available_dies, current_iteration + 1)
            if not sub_result:
                self._log(current_iteration, f"    Sub-region {i+1} has no solution")
                return None
            # Map sub-region coordinates back to original matrix space
            offset_mapping = self._apply_coordinate_offset(sub_result.mapping, spec)
            sub_mappings.append(offset_mapping)

        # Combine main mapping with sub-mappings
        combined_mapping = self._combine_mappings(main_mapping, sub_mappings)
        return self._validate_and_simulate(chip, combined_mapping)

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

    def _create_mapping_from_assignments(
        self,
        matrix: MatrixShape,
        chip: Chip,
        assignments: Dict[str, List[Tuple[int, int, int, int, int, int]]],
    ) -> Mapping:
        """Create a mapping object from tile assignments (legacy tuple format)."""
        mapping = Mapping(matrix=matrix, chip=chip)
        for die_id, tile_list in assignments.items():
            for r0, r1, c0, c1, b0, b1 in tile_list:
                num_rows = r1 - r0
                num_cols = c1 - c0
                num_batches = b1 - b0
                mapping.add_tile(die_id, num_rows, num_cols, num_batches)
        return mapping

    def _create_mapping_from_specs(
        self,
        matrix: MatrixShape,
        chip: Chip,
        assignments: Dict[str, List[SubMatrixSpec]],
    ) -> Mapping:
        """Create a mapping object from SubMatrixSpec assignments."""
        mapping = Mapping(matrix=matrix, chip=chip)
        for die_id, spec_list in assignments.items():
            for spec in spec_list:
                num_rows = spec.row1 - spec.row0
                num_cols = spec.col1 - spec.col0
                num_batches = spec.batch1 - spec.batch0
                mapping.add_tile(die_id, num_rows, num_cols, num_batches)
        return mapping

    def _construct_tail_subregions(
        self,
        row_bounds: List[Tuple[int, int]],
        col_bounds: List[Tuple[int, int]],
        remaining_tiles: List[SubMatrixSpec],
    ) -> List[SubMatrixSpec]:
        """Construct at most two rectangular subregions that exactly cover the tail.

        The tail in row-major order always consists of:
        - Zero or more complete bottom rows, and
        - An optional right-edge suffix of the row just above them.

        This geometric precision is critical for correctness.
        """
        if not remaining_tiles:
            return []

        # Extract batch bounds from remaining tiles (all should have same batch bounds)
        batch0 = remaining_tiles[0].batch0
        batch1 = remaining_tiles[0].batch1

        # Grid dimensions
        g_rows = len(row_bounds)
        g_cols = len(col_bounds)

        # Analyze tail structure
        num_tail = len(remaining_tiles)
        full_rows = num_tail // g_cols
        suffix_cols = num_tail % g_cols

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
                specs = []
                for r in regions:
                    spec = self._grid_rect_to_spec(row_bounds, col_bounds, *r, batch0, batch1)
                    if spec is not None:
                        specs.append(spec)
                return specs

        return []

    def _grid_rect_to_spec(
        self,
        row_bounds: List[Tuple[int, int]],
        col_bounds: List[Tuple[int, int]],
        r0_idx: int,
        r1_idx_excl: int,
        c0_idx: int,
        c1_idx_excl: int,
        batch0: int = 0,
        batch1: int = 1
    ) -> Optional[SubMatrixSpec]:
        """Convert grid indices to matrix coordinates."""
        if r0_idx >= r1_idx_excl or c0_idx >= c1_idx_excl:
            return None
        row0 = row_bounds[r0_idx][0]
        row1 = row_bounds[r1_idx_excl - 1][1]
        col0 = col_bounds[c0_idx][0]
        col1 = col_bounds[c1_idx_excl - 1][1]
        return SubMatrixSpec(row0=row0, row1=row1, col0=col0, col1=col1, batch0=batch0, batch1=batch1)

    def _apply_coordinate_offset(self, mapping: Mapping, spec: SubMatrixSpec) -> Mapping:
        """Copy sub-region mapping to original matrix space.

        Note: Since Tile no longer stores position information, we just copy the tiles
        with their dimensions. The spec parameter is kept for API compatibility but
        position offset is no longer needed.
        """
        offset_mapping = Mapping(matrix=mapping.matrix, chip=mapping.chip)
        for die_id, tiles in mapping.placement.items():
            for tile in tiles:
                offset_mapping.add_tile(
                    die_id,
                    tile.num_rows,
                    tile.num_cols,
                    tile.num_batches,
                )
        return offset_mapping

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
    def _memo_key(self, matrix: MatrixShape, available_dies: Set[str]) -> Tuple:
        """Create a memoization key for the subproblem."""
        return (matrix.rows, matrix.cols, matrix.batch_size, tuple(sorted(available_dies)))

    def _round_robin_fallback(
        self,
        matrix: MatrixShape,
        chip: Chip,
        main_assignments: Dict[str, List[SubMatrixSpec]],
        remaining_tiles: List[SubMatrixSpec]
    ) -> Optional[MappingResult]:
        """Fallback strategy: distribute remaining tiles using round-robin."""
        die_ids = sorted(list(main_assignments.keys()))

        # Add remaining tiles using round-robin
        for i, tile_spec in enumerate(remaining_tiles):
            die_id = die_ids[i % len(die_ids)]
            main_assignments[die_id].append(tile_spec)

        # Create and validate the mapping
        combined_mapping = self._create_mapping_from_specs(matrix, chip, main_assignments)
        return self._validate_and_simulate(chip, combined_mapping)