"""Matrix-level hardware mapping pass for computation graphs.

This pass traverses the computation graph and maps each matrix operation
(MatMulOp) to hardware channels using matrixmapper strategies.
"""

from typing import Optional
import math

from pimapper.modelmapper.passes.base import Pass
from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.hwspec import AcceleratorSpec
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType
from pimapper.matrixmapper.interface import create_mapping, StrategyType


class MatrixMappingPass(Pass):
    """Pass that maps matrix operations to hardware channels.

    This pass:
    1. Traverses all nodes in the computation graph
    2. Identifies matrix operations (MatMulOp)
    3. Extracts matrix dimensions and batch information
    4. Uses matrixmapper.create_mapping() to find optimal hardware mapping
    5. Stores the MappingResult in op.kwargs['mapping_result']

    Args:
        accelerator_spec: Hardware specification for the accelerator
        strategy: Mapping strategy ("trivial", "recursive_grid_search", "h2llm")
        strategy_kwargs: Optional configuration dict for the strategy
        name: Optional custom name for the pass
    """

    def __init__(
        self,
        accelerator_spec: AcceleratorSpec,
        strategy: StrategyType = "recursive_grid_search",
        strategy_kwargs: Optional[dict] = None,
        *,
        name: Optional[str] = None
    ):
        super().__init__(
            name=name or "MatrixMappingPass",
            description="Maps matrix operations to hardware channels using matrixmapper strategies"
        )
        self.accelerator_spec = accelerator_spec
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}

    def _map_batched_matmul(self, op, node_name: str, data_format: DataFormat) -> Optional[float]:
        """Calculate latency for BatchedMatMulOp.

        Args:
            op: BatchedMatMulOp operation
            node_name: Name of the node
            data_format: DataFormat from inference_config

        Returns:
            Total latency for the batched matmul operation, or None if mapping fails
        """
        # Extract information from op.kwargs
        num_matmuls = op.kwargs.get('num_matmuls', 1)
        matmul_shape = op.kwargs.get('matmul_shape')

        if matmul_shape is None or len(matmul_shape) != 3:
            return None

        M, rows, cols = matmul_shape

        matrix_shape = MatrixShape(
            rows=rows,
            cols=cols,
            batch_size=M,
            data_format=data_format
        )

        # Create single-channel accelerator spec
        from pimapper.core.hwspec import AcceleratorSpec
        single_channel_spec = AcceleratorSpec(
            channel_count=1,
            channel_spec=self.accelerator_spec.channel_spec
        )

        # Map single matmul to get latency
        mapping_result = create_mapping(
            matrix_shape=matrix_shape,
            accelerator_spec=single_channel_spec,
            strategy=self.strategy,
            **self.strategy_kwargs
        )

        if mapping_result is None:
            return None

        single_latency = mapping_result.latency

        # Calculate number of rounds needed
        num_channels = self.accelerator_spec.channel_count
        num_rounds = math.ceil(num_matmuls / num_channels)

        # Total latency = single latency * number of rounds
        total_latency = single_latency * num_rounds

        return total_latency

    def _extract_matrix_info(self, op, inference_batch_size: int) -> Optional[tuple[int, int, int]]:
        """Extract matrix dimensions from an operation.

        Args:
            op: Operation object to extract info from
            inference_batch_size: Batch size from inference_config

        Returns:
            Tuple of (rows, cols, batch_size) or None if extraction fails
        """
        # For FusionMatrixOp, use the fused_weight_shape directly
        if hasattr(op, 'fused_weight_shape') and op.fused_weight_shape is not None:
            rows, cols = op.fused_weight_shape
            return rows, cols, inference_batch_size

        # For regular MatMulOp, extract weight shape from metadata
        if hasattr(op, 'metadata') and op.metadata:
            custom = op.metadata.get('custom', {})
            weight_shape = custom.get('weight_shape')

            if weight_shape is not None:
                # weight_shape is (out_features, in_features) for Linear layers
                # Check if transpose_b is True (which means weight is transposed in matmul)
                transpose_b = op.kwargs.get('transpose_b', False) if hasattr(op, 'kwargs') else False

                if transpose_b:
                    # Weight is transposed: (out_features, in_features) -> (in_features, out_features)
                    rows, cols = weight_shape[1], weight_shape[0]
                else:
                    rows, cols = weight_shape

                return rows, cols, inference_batch_size

        return None

    def _map_matrix_op(self, op, node_name: str, data_format: DataFormat, inference_batch_size: int) -> Optional[float]:
        """Calculate latency for regular MatMulOp or FusionMatrixOp.

        Args:
            op: MatMulOp or FusionMatrixOp operation
            node_name: Name of the node
            data_format: DataFormat from inference_config
            inference_batch_size: Batch size from inference_config

        Returns:
            Latency for the matrix operation, or None if mapping fails
        """
        # Extract matrix information
        matrix_info = self._extract_matrix_info(op, inference_batch_size)
        if matrix_info is None:
            return None

        rows, cols, batch_size = matrix_info

        # Create MatrixShape
        matrix_shape = MatrixShape(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            data_format=data_format
        )

        # Find optimal mapping using create_mapping interface
        mapping_result = create_mapping(
            matrix_shape=matrix_shape,
            accelerator_spec=self.accelerator_spec,
            strategy=self.strategy,
            **self.strategy_kwargs
        )

        if mapping_result is None:
            return None

        # Store mapping result in op.kwargs
        if not hasattr(op, 'kwargs'):
            op.kwargs = {}
        op.kwargs['mapping_result'] = mapping_result
        op.kwargs['latency'] = mapping_result.latency

        return mapping_result.latency

    def run(self, graph: NxComputationGraph) -> bool:
        """Execute the matrix mapping pass on the computation graph.

        Args:
            graph: Computation graph to process

        Returns:
            True if any matrices were mapped, False otherwise

        Raises:
            RuntimeError: If a matrix operation cannot be mapped or inference_config is missing
        """
        # Initialize metadata
        self._metadata = {
            "matrices_mapped": 0,
            "total_latency": 0.0,
            "failed_mappings": [],
            "mapping_details": []
        }

        # Read inference_config from graph metadata
        inference_config_meta = graph.metadata.get('inference_config', {})
        data_format = inference_config_meta.get('data_format')
        batch_size = inference_config_meta.get('batch_size', 1)

        # Validate that data_format is provided
        if data_format is None:
            raise RuntimeError(
                "inference_config.data_format is required in graph metadata. "
                "Please ensure the graph has been configured with a DataFormat."
            )

        modified = False

        # Traverse all nodes in the graph
        for node_name in graph.nodes(sort=True):
            op = graph.node_record(node_name)

            # Check if this is a matrix operation
            op_type = getattr(op, 'op_type', None)
            if op_type not in ("matmul", "fusion_matrix", "batched_matmul"):
                continue

            # Handle batched_matmul
            if op_type == "batched_matmul":
                latency = self._map_batched_matmul(op, node_name, data_format)

                if latency is None:
                    error_msg = f"Failed to map batched_matmul operation '{node_name}'"
                    self._metadata["failed_mappings"].append(node_name)
                    raise RuntimeError(error_msg)

                # Store latency in op.kwargs
                if not hasattr(op, 'kwargs'):
                    op.kwargs = {}
                op.kwargs['latency'] = latency

                # Update metadata
                self._metadata["matrices_mapped"] += 1
                self._metadata["total_latency"] += latency
                self._metadata["mapping_details"].append({
                    "node_name": node_name,
                    "op_type": "batched_matmul",
                    "latency": latency
                })

                modified = True
                continue

            # Handle regular matmul and fusion_matrix
            latency = self._map_matrix_op(op, node_name, data_format, batch_size)

            if latency is None:
                error_msg = f"Failed to map matrix operation '{node_name}'"
                self._metadata["failed_mappings"].append(node_name)
                raise RuntimeError(error_msg)

            # Get matrix info for metadata
            matrix_info = self._extract_matrix_info(op, batch_size)
            if matrix_info is not None:
                rows, cols, batch = matrix_info
                mapping_result = op.kwargs.get('mapping_result')

                self._metadata["mapping_details"].append({
                    "node_name": node_name,
                    "shape": (rows, cols, batch),
                    "latency": latency,
                    "utilization": mapping_result.get_compute_utilization() if mapping_result else None
                })

            # Update metadata
            self._metadata["matrices_mapped"] += 1
            self._metadata["total_latency"] += latency

            modified = True

        return modified
