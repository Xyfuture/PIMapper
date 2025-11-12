"""H2-LLM Mapping Strategy Implementation.

Based on the paper: H2-LLM: Hardware-Dataflow Co-Exploration for Heterogeneous
Hybrid-Bonding-based Low-Batch LLM Inference (ISCA 2025)

This module implements the inter-channel operator partition algorithm described
in Section 4.2 of the H2-LLM paper. The key innovation is using an analytical
model to find optimal tiling factors that minimize data transfer overhead.

Key algorithm from paper (Section 4.2):
    For GEMM operators with input shape (M, K) and weight shape (K, N), the
    optimal tiling factors T_K and T_N across C channels are determined by
    minimizing the total transfer overhead:

    min_{T_K, T_N} s × M × (K/(T_K × B_s) + N/(T_N × B_l))
    s.t. T_K × T_N = C

    Where:
    - s: element size (bytes)
    - M, K, N: matrix dimensions
    - C: number of NMP channels
    - B_l: load (input) bandwidth per channel
    - B_s: store (output) bandwidth per channel

    Analytical solution: T_K = sqrt(C × K × B_s / (N × B_l))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Tuple

from ...core.hwspec import Accelerator
from ...core.matrixspec import Mapping, MatrixShape
from ..evaluator import evaluate
from ...core.utils import MappingResult
from ..utils import MatrixAllocationTree

logger = logging.getLogger("h2llm_mapping")


@dataclass
class H2LLMTilingStrategy:
    """H2-LLM's inter-channel operator partition strategy.

    This strategy implements the analytical model from the H2-LLM paper to
    determine optimal tiling factors for matrix operations across NMP channels.

    The algorithm:
    1. Extract bandwidth parameters from chip's compute die specifications
    2. Calculate optimal K and N dimension tiling factors using the analytical model
    3. Split the matrix according to these factors
    4. Distribute tiles across compute dies
    5. Validate and simulate the mapping

    Attributes:
        element_size: Size of each matrix element in bytes (default: 2 for FP16)

    Note:
        Bandwidth parameters are automatically extracted from the chip specification:
        - load_bandwidth: chip.spec.die_spec.get_input_bandwidth()
        - store_bandwidth: chip.spec.die_spec.get_output_bandwidth()

        These methods handle both separate bandwidths and shared_bandwidth modes.
    """

    element_size: float = field(default=2.0)  # bytes (FP16)

    def create_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
    ) -> Tuple[Mapping, MatrixAllocationTree]:
        """Create a mapping using H2-LLM's analytical tiling strategy.

        Args:
            matrix_shape: Matrix dimensions (M × N with batch_size)
            accelerator: Hardware accelerator configuration

        Returns:
            Tuple of (Mapping object with optimized tile assignments, MatrixAllocationTree)

        Raises:
            ValueError: If parameters are invalid
        """
        if not accelerator.channels:
            raise ValueError("Accelerator must have at least one PIM channel")
        if matrix_shape.rows <= 0 or matrix_shape.cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
        if matrix_shape.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Extract bandwidth parameters from the accelerator's channel spec
        # (assuming all channels have the same configuration)
        channel_spec = accelerator.spec.channel_spec
        load_bandwidth = channel_spec.get_input_bandwidth()  # GB/s
        store_bandwidth = channel_spec.get_output_bandwidth()    # GB/s

        M = 0
        K = matrix_shape.rows
        N = matrix_shape.cols
        batch_size = matrix_shape.batch_size
        C = len(accelerator.channels)

        logger.debug(f"H2-LLM Mapping: Matrix {M}×{N}×{batch_size}, {C} PIM channels")
        logger.debug(f"  Load BW: {load_bandwidth} GB/s, Store BW: {store_bandwidth} GB/s")

        # For GEMM: Input (M, K) × Weight (K, N) = Output (M, N)
        # We treat the matrix as M×N, where:
        # - M is the batch dimension (not split to avoid weight duplication)
        # - K would be the reduction dimension
        # - N is the output feature dimension (can be split)

        # Calculate optimal tiling factors using H2-LLM's analytical model
        # T_K, T_N = self._calculate_optimal_tiling(M, K, N, C, load_bandwidth, store_bandwidth)
        T_K, T_N = self._calculate_optimal_tiling_h2_paper(M, K, N, C, load_bandwidth, store_bandwidth)

        logger.debug(f"  Optimal tiling factors: T_K={T_K}, T_N={T_N}")
        logger.debug(f"  This creates {T_K * T_N} tiles across {C} channels")

        # Create the mapping with calculated tiling
        mapping, tree = self._create_tiled_mapping(matrix_shape, accelerator, T_K, T_N)

        # Validate the mapping
        try:
            mapping.check_all()
            logger.debug("  Mapping validation successful")
        except Exception as e:
            logger.error(f"  Mapping validation failed: {e}")
            raise

        return mapping, tree

    def find_optimal_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
    ) -> Optional[MappingResult]:
        """Find the optimal mapping and evaluate its performance.

        This method creates the mapping and evaluates it to get the latency.

        Args:
            matrix_shape: Matrix dimensions
            accelerator: Hardware accelerator configuration

        Returns:
            MappingResult with mapping and latency, or None if mapping fails
        """
        try:
            mapping, tree = self.create_mapping(matrix_shape, accelerator)
            latency = evaluate(accelerator, mapping)

            result = MappingResult(mapping=mapping, latency=latency, allocation_tree=tree)
            utilization = result.get_compute_utilization()

            logger.debug(f"  Simulation complete: latency={latency:.2f}, utilization={utilization:.2%}")

            return result
        except Exception as e:
            logger.error(f"  Failed to create optimal mapping: {e}")
            return None

    def _calculate_optimal_tiling(
        self,
        M: int,
        K: int,
        N: int,
        C: int,
        load_bandwidth: float,
        store_bandwidth: float,
    ) -> Tuple[int, int]:
        """Calculate optimal tiling factors using H2-LLM's analytical model.

        From the paper (Section 4.2):
        The optimal tiling factors minimize:
            s × M × (K/(T_K × B_l) + N/(T_N × B_s))
        subject to: T_K × T_N = C

        Analytical solution:
            T_K = sqrt(C × K × B_s / (N × B_l))
            T_N = C / T_K

        Args:
            M: Batch/row dimension (not split in H2-LLM to avoid weight duplication)
            K: Reduction dimension
            N: Output feature dimension
            C: Number of compute channels
            load_bandwidth: Input/load bandwidth per channel (B_l)
            store_bandwidth: Output/store bandwidth per channel (B_s)

        Returns:
            Tuple of (T_K, T_N) tiling factors
        """
        B_l = load_bandwidth
        B_s = store_bandwidth

        # Analytical solution from the paper
        if N * B_l > 0:
            T_K_float = sqrt(C * K * B_s / (N * B_l))
        else:
            T_K_float = sqrt(C)

        # Round to nearest integer and ensure valid range
        T_K = max(1, min(C, round(T_K_float)))
        T_N = max(1, C // T_K)

        # Adjust to ensure T_K * T_N <= C (don't over-partition)
        while T_K * T_N > C and T_K > 1:
            T_K -= 1
            T_N = C // T_K

        # If we still can't fit, try the other way
        if T_K * T_N > C:
            T_N = max(1, C // T_K)

        # Ensure at least some partitioning if C > 1
        if T_K * T_N < C and C > 1:
            if T_N * (T_K + 1) <= C:
                T_K += 1
            elif T_K * (T_N + 1) <= C:
                T_N += 1

        assert T_K * T_N == C,  "Wrong dim divisor allocation"


        logger.debug(f"    Analytical T_K={T_K_float:.2f} -> rounded T_K={T_K}, T_N={T_N}")

        return T_K, T_N

    def _calculate_optimal_tiling_h2_paper(
        self,
        M: int,
        K: int,
        N: int,
        C: int,
        load_bandwidth: float,
        store_bandwidth: float,
    ) -> Tuple[int, int]:
        """Calculate optimal tiling factors using H2-LLM's original algorithm from nmp_evaluator.py.

        This implements the exact algorithm from the H2-LLM paper's evaluator code,
        which uses a different approach based on GCD calculation and factor analysis.

        The algorithm:
        1. Assume batch size = 1 (always B == 1 case)
        2. Find optimal K dimension divisor using factor analysis
        3. Calculate corresponding N dimension divisor

        Args:
            M: M dimension (rows, batch dimension - not used in calculation)
            K: K dimension (reduction)
            N: N dimension (columns)
            C: Number of compute channels
            load_bandwidth: Input/load bandwidth per channel (not used in original algorithm)
            store_bandwidth: Output/store bandwidth per channel (not used in original algorithm)

        Returns:
            Tuple of (K_dim_divisor, N_dim_divisor) tiling factors
        """


        per_gemm_channel_num = C

        # Step 2: Find optimal K dimension divisor using factor analysis (line 67)
        optimal_K_dim_divisor = self._find_closest_factor(
            per_gemm_channel_num,
            math.sqrt(K * per_gemm_channel_num*load_bandwidth / (N*store_bandwidth))
        )

        # Step 3: Calculate corresponding N dimension divisor (line 68)
        optimal_N_dim_divisor = per_gemm_channel_num // optimal_K_dim_divisor

        # Validate the allocation (line 69)
        assert optimal_K_dim_divisor * optimal_N_dim_divisor == per_gemm_channel_num, \
            "Wrong dim divisor allocation"

        logger.debug(f"    H2-paper algorithm: per_gemm_channel_num={per_gemm_channel_num}")
        logger.debug(f"    Load BW: {load_bandwidth}, Store BW: {store_bandwidth} (not used in original)")
        logger.debug(f"    Optimal divisors: K={optimal_K_dim_divisor}, N={optimal_N_dim_divisor}")

        return optimal_K_dim_divisor, optimal_N_dim_divisor

    def _find_closest_factor(self, n: int, target: float) -> int:
        """Find the factor of n that is closest to the target value.

        This implements the get_factors and find_closest functions from nmp_evaluator.py.

        Args:
            n: The number to find factors of
            target: The target value to find the closest factor to

        Returns:
            The factor of n that is closest to target
        """
        # Get all factors of n (lines 7-13 from nmp_evaluator.py)
        factors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)

        # Find the factor closest to target (line 16-17 from nmp_evaluator.py)
        return min(factors, key=lambda x: abs(x - target))

    def _create_tiled_mapping(
        self,
        matrix_shape: MatrixShape,
        accelerator: Accelerator,
        T_K: int,
        T_N: int,
    ) -> Tuple[Mapping, MatrixAllocationTree]:
        """Create a mapping with specified tiling factors.

        The tiling strategy:
        - Split rows into T_K parts (for K dimension)
        - Split cols into T_N parts (for N dimension)
        - Split batch dimension evenly across PIM channels if needed
        - Distribute tiles to channels in round-robin fashion

        Args:
            matrix_shape: Matrix dimensions
            accelerator: Accelerator configuration
            T_K: Tiling factor for row dimension
            T_N: Tiling factor for column dimension

        Returns:
            Tuple of (Mapping object with tile assignments, MatrixAllocationTree)
        """
        # Calculate tile boundaries
        row_bounds = self._split_dimension(matrix_shape.rows, T_K)
        col_bounds = self._split_dimension(matrix_shape.cols, T_N)

        # Do not split batch dimension to avoid weight duplication (H2-LLM paper principle)
        batch_bounds = [(0, matrix_shape.batch_size)]
        num_channels = len(accelerator.channels)

        logger.debug(f"    Row splits: {len(row_bounds)}, Col splits: {len(col_bounds)}, Batch splits: 1 (no split)")

        # Create mapping
        mapping = Mapping(matrix=matrix_shape, accelerator=accelerator)
        channel_ids = sorted(accelerator.channels.keys())

        # Create allocation tree
        tree = MatrixAllocationTree.create_root(
            rows=matrix_shape.rows,
            cols=matrix_shape.cols,
            batch_size=matrix_shape.batch_size,
            num_split_row=len(row_bounds),
            num_split_col=len(col_bounds),
            channel_ids=channel_ids
        )

        # Calculate total tiles (batch is not split, so only spatial tiles)
        total_tiles = len(row_bounds) * len(col_bounds)

        # Add leaf child with all tiles (single-level tree)
        tree.root.add_leaf_child(num_tiles=total_tiles)

        # Distribute tiles in row-major order with round-robin assignment
        tile_idx = 0
        for r0, r1 in row_bounds:
            for c0, c1 in col_bounds:
                for b0, b1 in batch_bounds:
                    channel_id = channel_ids[tile_idx % num_channels]
                    # add_tile expects dimensions (num_rows, num_cols, num_batches), not boundaries
                    mapping.add_tile(channel_id, r1 - r0, c1 - c0, b1 - b0)
                    tile_idx += 1

        logger.debug(f"    Created {tile_idx} tiles distributed across {num_channels} channels")

        # Assign tile IDs to the tree
        success = tree.assign_tile_ids()
        if not success:
            logger.warning("Failed to assign tile IDs in H2LLM mapping")

        return mapping, tree

    def _split_dimension(self, total: int, parts: int) -> List[Tuple[int, int]]:
        """Split a dimension into parts with balanced sizes.

        Uses ceiling division to handle remainders, similar to the paper's
        approach for handling non-divisible dimensions.

        Args:
            total: Total dimension size
            parts: Number of parts to split into

        Returns:
            List of (start, end) tuples defining half-open intervals
        """
        if parts <= 0:
            return [(0, total)]
        if parts > total:
            parts = total

        base = total // parts
        extra = total % parts

        bounds: List[Tuple[int, int]] = []
        start = 0

        for idx in range(parts):
            # Give extra elements to the first 'extra' parts
            length = base + (1 if idx < extra else 0)
            end = start + length
            bounds.append((start, end))
            start = end

        return bounds

