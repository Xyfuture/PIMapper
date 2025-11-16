from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ...core.hwspec import Accelerator
from ...core.matrixspec import Mapping, MatrixShape
from ..evaluator import evaluate
from ...core.utils import MappingResult
from ..utils import MatrixAllocationTree, InternalNode

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

    def _validate_and_evaluate(
        self,
        accelerator: Accelerator,
        mapping: Mapping,
        allocation_tree: Optional[MatrixAllocationTree] = None
    ) -> Optional[MappingResult]:
        """Validate mapping and run evaluation, return None on failure."""
        try:
            mapping.check_all()
            latency = evaluate(accelerator, mapping)
            return MappingResult(mapping=mapping, latency=latency, allocation_tree=allocation_tree)
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

        # 在最后一次递归时，添加特殊候选以保证能整除
        if current_iteration == self.max_iterations:
            n_channels = len(available_channels)
            if 1 <= n_channels <= matrix_shape.cols and (1, n_channels) not in split_candidates:
                split_candidates.append((1, n_channels))
            if 1 <= n_channels <= matrix_shape.rows and (n_channels, 1) not in split_candidates:
                split_candidates.append((n_channels, 1))

        # Try all split configurations
        split_count = 0
        for num_r, num_c in split_candidates:
            # Skip invalid configurations
            if not self._is_valid_split(num_r, num_c, matrix_shape):
                continue

            # 在最后一次递归时，只考虑能整除的配置
            n_tiles = num_r * num_c
            if current_iteration == self.max_iterations and n_tiles % len(available_channels) != 0:
                self._log(current_iteration, f"  Skipping {num_r}x{num_c}: {n_tiles} tiles not divisible by {len(available_channels)} channels")
                continue

            split_count += 1
            self._log(current_iteration, f"  Trying split config {split_count}: {num_r}x{num_c} (rows x cols)")

            # Pruning: Check utilization upper bound
            if best:
                # estimated_util = self._estimate_utilization(accelerator, matrix_shape, num_r, num_c)
                # current_best_util = best.get_compute_utilization()
                estimated_latency = self._estimate_latency(accelerator,matrix_shape,num_r,num_c)
                cur_best_latency = best.latency

                # print(f"  Estimated utilization: {estimated_util:.2%} Current best: {current_best_util:.2%}"   )
                if estimated_latency > cur_best_latency:
                    self._log(current_iteration, f"    Pruned: estimated latency {estimated_latency:.4f} < best latency {cur_best_latency:.4f}")
                    # print(f"    Pruned: estimated latency {estimated_latency:.4f} < best latency {cur_best_latency:.4f}"   )
                    continue

            try:
                candidate = self._evaluate_split_configuration(
                    matrix_shape, accelerator, available_channels, num_r, num_c, current_iteration
                )
            except Exception as e:
                self._log(current_iteration, f"    Configuration failed: {type(e).__name__}: {e}")
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

            # Validate tree structure at root level
            if current_iteration == 0 and best.allocation_tree is not None:
                tree_valid = best.allocation_tree.validate()
                if tree_valid:
                    self._log(current_iteration, "  Tree structure validated successfully")
                else:
                    logger.warning("Tree validation failed")
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
        current_tree_node: Optional[InternalNode] = None,
    ) -> Optional[MappingResult]:
        """Evaluate a specific split configuration and return the mapping result with allocation tree."""

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

        # Create or use tree node for this split
        if current_tree_node is None:
            # This is the root - create the tree
            tree = MatrixAllocationTree.create_root(
                rows=matrix.rows,
                cols=matrix.cols,
                batch_size=matrix.batch_size,
                num_split_row=num_split_row,
                num_split_col=num_split_col
            )
            current_node = tree.root
        else:
            # This is a child node - use the provided node
            current_node = current_tree_node
            tree = None

        # Progress guarantee: if we have enough channels, assign all tiles
        if n_tiles <= n_channels:
            self._log(current_iteration, f"    Tiles <= channels, direct assignment")
            assignments: Dict[str, List[MatrixShape]] = {c: [] for c in channel_ids}
            for i, tile in enumerate(tiles):
                assignments[channel_ids[i % n_channels]].append(tile)

            # Add leaf node to tree
            current_node.add_leaf_child(num_tiles=n_tiles)

            main_mapping = self._create_mapping_from_specs(matrix, accelerator, assignments)

            # Don't assign tile IDs during recursion - only build tree structure
            return self._validate_and_evaluate(accelerator, main_mapping, tree)

        # Round-robin distribution: complete rounds only
        tiles_per_channel = n_tiles // n_channels
        assigned_count = tiles_per_channel * n_channels

        self._log(current_iteration, f"    Round-robin: {tiles_per_channel} tiles per channel, total {assigned_count} assigned")

        main_assignments: Dict[str, List[MatrixShape]] = {c: [] for c in channel_ids}
        for i in range(assigned_count):
            main_assignments[channel_ids[i % n_channels]].append(tiles[i])

        remaining_tiles = tiles[assigned_count:]
        main_mapping = self._create_mapping_from_specs(matrix, accelerator, main_assignments)

        # Add leaf node for assigned tiles
        current_node.add_leaf_child(num_tiles=assigned_count)

        # If no remaining tiles, we're done
        if not remaining_tiles:
            self._log(current_iteration, f"    No remaining tiles, assignment complete")

            # Don't assign tile IDs during recursion - only build tree structure
            return self._validate_and_evaluate(accelerator, main_mapping, tree)

        self._log(current_iteration, f"    Remaining {len(remaining_tiles)} tiles for recursive processing")

        # Check iteration limit before recursion
        if current_iteration >= self.max_iterations:
            self._log(current_iteration, f"    Max recursion depth ({self.max_iterations}) reached, using round-robin fallback")
            # Fallback: Use round-robin distribution for remaining tiles
            for i, tile in enumerate(remaining_tiles):
                main_assignments[channel_ids[i % n_channels]].append(tile)

            fallback_mapping = self._create_mapping_from_specs(matrix, accelerator, main_assignments)
            result = self._validate_and_evaluate(accelerator, fallback_mapping, tree)
            return result

        # Handle tail tiles by constructing decomposition plan
        sub_regions = self._construct_tail_subregions(row_bounds, col_bounds, len(remaining_tiles), matrix.batch_size)

        if not sub_regions:
            self._log(current_iteration, f"    Failed to construct tail subregions")
            return None

        self._log(current_iteration, f"    Decomposed into {len(sub_regions)} tail sub-regions")

        # Recursively solve each subregion
        sub_mappings: List[Mapping] = []

        for i, (sub_shape, num_parent_tiles) in enumerate(sub_regions):
            self._log(current_iteration, f"      Sub-region {i+1}: {sub_shape.rows}x{sub_shape.cols}x{sub_shape.batch_size}, occupies {num_parent_tiles} parent tiles")

            # Create internal child node for this tail region
            child_node = current_node.add_internal_child(
                rows=sub_shape.rows,
                cols=sub_shape.cols,
                batch_size=sub_shape.batch_size,
                num_split_row=1,  # Will be updated by recursive call
                num_split_col=1,
                num_parent_tiles=num_parent_tiles
            )

            # Recursively solve this tail region
            sub_result = self.find_optimal_mapping(
                sub_shape,
                accelerator,
                available_channels,
                current_iteration + 1
            )

            if not sub_result:
                self._log(current_iteration, f"      Sub-region {i+1} has no solution")
                return None

            # Copy the tree structure from sub_result to child_node
            if sub_result.allocation_tree:
                child_node.num_split_row = sub_result.allocation_tree.root.num_split_row
                child_node.num_split_col = sub_result.allocation_tree.root.num_split_col
                child_node.leaf_child = sub_result.allocation_tree.root.leaf_child
                child_node.internal_children = sub_result.allocation_tree.root.internal_children

            sub_mappings.append(self._copy_mapping(sub_result.mapping))

        # Combine main mapping with sub-mappings
        combined_mapping = self._combine_mappings(main_mapping, sub_mappings)
        result = self._validate_and_evaluate(accelerator, combined_mapping, tree)

        return result

    # ---------------
    # Geometry and tiling helpers
    # ---------------
    def _factorize(self, n: int) -> List[Tuple[int, int]]:
        """返回 n 的所有因数对 (r, c)，满足 r * c == n"""
        factors = []
        for r in range(1, int(n**0.5) + 1):
            if n % r == 0:
                c = n // r
                factors.append((r, c))
                if r != c:
                    factors.append((c, r))
        return factors

    def _calculate_split_boundaries(self, dimension: int, num_splits: int) -> List[Tuple[int, int]]:
        """Calculate split boundaries with ceiling operation for edge tiles."""
        if num_splits <= 0:
            return [(0, dimension)]

        base_size = dimension // num_splits
        base_size = int(dimension/num_splits + 0.5)
        remainder = dimension % num_splits

        boundaries = []
        start = 0

        # for i in range(num_splits):
        #     # Distribute remainder to first `remainder` splits (ceiling operation)
        #     size = base_size + (1 if i < remainder else 0)
        #     end = start + size
        #     boundaries.append((start, end))
        #     start = end
        for i in range(num_splits):
            end = start + base_size + 1 
            boundaries.append((start,end))
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
    ) -> List[Tuple[MatrixShape, int]]:
        """返回单一确定性的分解方案，包含1-2个子区域。

        Returns:
            List of (MatrixShape, num_tiles) tuples, where num_tiles is the number of
            parent grid tiles this sub-region occupies.
        """
        if num_remaining_tiles == 0:
            return []

        g_rows = len(row_bounds)
        g_cols = len(col_bounds)

        # 尝试单矩阵方案：找到能整除的因数分解
        for r, c in self._factorize(num_remaining_tiles):
            if r <= g_rows and c <= g_cols:
                shape = self._grid_rect_to_shape(
                    row_bounds, col_bounds,
                    g_rows - r, g_rows,
                    g_cols - c, g_cols,
                    batch_size
                )
                if shape is not None:
                    return [(shape, num_remaining_tiles)]

        # Fallback：两矩阵方案（类似 ref_recursive.py）
        full_rows = num_remaining_tiles // g_cols
        suffix_cols = num_remaining_tiles % g_cols

        if suffix_cols > 0 and full_rows > 0:
            # 部分行 + 完整行
            regions = [
                (g_rows - full_rows - 1, g_rows - full_rows, g_cols - suffix_cols, g_cols, suffix_cols),
                (g_rows - full_rows, g_rows, 0, g_cols, full_rows * g_cols)
            ]
            shapes = []
            for r0, r1, c0, c1, n_tiles in regions:
                shape = self._grid_rect_to_shape(row_bounds, col_bounds, r0, r1, c0, c1, batch_size)
                if shape is not None:
                    shapes.append((shape, n_tiles))
            if len(shapes) == 2:
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
        copied_mapping = Mapping(matrix=mapping.matrix, accelerator=mapping.accelerator)
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
        combined = Mapping(matrix=main_mapping.matrix, accelerator=main_mapping.accelerator)

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

    def _estimate_latency(
            self,
            accelerator: Accelerator,
            matrix: MatrixShape,
            num_row_splits: int,
            num_col_splits: int    
        )->float:
        channel_spec = accelerator.spec.channel_spec
        batch_size = matrix.batch_size
        total_input_size = matrix.rows* num_row_splits * matrix.data_format.input_dtype.bytes_per_element  * batch_size
        total_output_size = matrix.cols * num_row_splits * matrix.data_format.output_dtype.bytes_per_element * batch_size

        total_ops = batch_size * matrix.cols * matrix.rows 

        io_latency = (total_input_size + total_output_size) / (channel_spec.shared_bandwidth * accelerator.spec.channel_count)

        compute_latency = total_ops / (channel_spec.compute_power*1000 * accelerator.spec.channel_count)

        return max(io_latency,compute_latency)

            
        
