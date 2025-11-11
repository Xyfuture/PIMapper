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

    def _torch_dtype_to_datatype(self, torch_dtype: Optional[str]) -> DataType:
        """Convert torch dtype string to DataType enum.

        Args:
            torch_dtype: String like "torch.float32", "torch.float16", etc.

        Returns:
            DataType enum value (defaults to FP16 if unknown)
        """
        if torch_dtype is None:
            return DataType.FP16

        dtype_lower = torch_dtype.lower()

        if "float16" in dtype_lower or "fp16" in dtype_lower or "half" in dtype_lower:
            return DataType.FP16
        elif "int8" in dtype_lower:
            return DataType.INT8
        elif "int4" in dtype_lower:
            return DataType.INT4
        elif "fp8" in dtype_lower or "float8" in dtype_lower:
            return DataType.FP8
        else:
            # Default to FP16 for float32 and other types
            return DataType.FP16

    def _extract_matrix_info(self, op) -> Optional[tuple[int, int, int, DataType]]:
        """Extract matrix dimensions and dtype from an operation.

        Args:
            op: Operation object to extract info from

        Returns:
            Tuple of (rows, cols, batch_size, dtype) or None if extraction fails
        """
        # For FusionMatrixOp, use the fused_weight_shape directly
        if hasattr(op, 'fused_weight_shape') and op.fused_weight_shape is not None:
            rows, cols = op.fused_weight_shape

            # Get batch size from output shape
            batch_size = 1
            if hasattr(op, 'results') and op.results:
                result = op.results[0]
                shape = getattr(result, 'shape', None)
                if shape and len(shape) > 2:
                    # For matrix operations, output shape is [batch_dims..., output_features]
                    # So batch dimensions are all except the last one
                    batch_dims = shape[:-1]
                    batch_size = math.prod(batch_dims)
                elif shape and len(shape) == 2:
                    # 2D output: first dimension is batch size
                    batch_size = shape[0]

                # Get dtype
                torch_dtype = getattr(result, 'dtype', None)
                dtype = self._torch_dtype_to_datatype(torch_dtype)
            else:
                dtype = DataType.FP16  # Default dtype

            return rows, cols, batch_size, dtype

        # Check if op has results
        if not hasattr(op, 'results') or not op.results:
            return None

        # Get first result (output tensor)
        result = op.results[0]

        # Get dtype
        torch_dtype = getattr(result, 'dtype', None)
        dtype = self._torch_dtype_to_datatype(torch_dtype)

        # Try to extract weight shape from metadata (for Linear layers)
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

                # Get batch size from output shape
                output_shape = getattr(result, 'shape', None)
                if output_shape and len(output_shape) > 2:
                    batch_dims = output_shape[:-1]  # All dimensions except the last one
                    batch_size = math.prod(batch_dims)
                elif output_shape and len(output_shape) == 2:
                    batch_size = output_shape[0]
                else:
                    batch_size = 1

                return rows, cols, batch_size, dtype

        # Fallback: extract from output shape (less accurate for batched operations)
        shape = getattr(result, 'shape', None)
        if shape is None or len(shape) < 2:
            return None

        # Extract batch and matrix dimensions
        if len(shape) == 2:
            # 2D matrix: no batch dimension
            rows, cols = shape
            batch_size = 1
        else:
            # 3D+ tensor: batch dimensions are all except last 2
            batch_dims = shape[:-2]
            matrix_dims = shape[-2:]

            # Calculate total batch size (product of all batch dimensions)
            batch_size = math.prod(batch_dims)
            rows, cols = matrix_dims

        return rows, cols, batch_size, dtype

    def run(self, graph: NxComputationGraph) -> bool:
        """Execute the matrix mapping pass on the computation graph.

        Args:
            graph: Computation graph to process

        Returns:
            True if any matrices were mapped, False otherwise

        Raises:
            RuntimeError: If a matrix operation cannot be mapped
        """
        # Initialize metadata
        self._metadata = {
            "matrices_mapped": 0,
            "total_latency": 0.0,
            "failed_mappings": [],
            "mapping_details": []
        }

        modified = False

        # Traverse all nodes in the graph
        for node_name in graph.nodes(sort=True):
            op = graph.node_record(node_name)

            # Check if this is a matrix operation (matmul or fusion_matrix)
            op_type = getattr(op, 'op_type', None)
            if op_type not in ("matmul", "fusion_matrix"):
                continue

            # Extract matrix information
            matrix_info = self._extract_matrix_info(op)
            if matrix_info is None:
                # Skip if we can't extract matrix info
                continue

            rows, cols, batch_size, dtype = matrix_info

            # Create DataFormat
            data_format = DataFormat(
                input_dtype=dtype,
                output_dtype=dtype,
                weight_dtype=dtype
            )

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

            # Error handling: raise exception if mapping fails
            if mapping_result is None:
                error_msg = (
                    f"Failed to map matrix operation '{node_name}' "
                    f"with shape ({rows}, {cols}, batch={batch_size})"
                )
                self._metadata["failed_mappings"].append(node_name)
                raise RuntimeError(error_msg)

            # Store mapping result in op.kwargs
            if not hasattr(op, 'kwargs'):
                op.kwargs = {}
            op.kwargs['mapping_result'] = mapping_result

            # Update metadata
            self._metadata["matrices_mapped"] += 1
            self._metadata["total_latency"] += mapping_result.latency
            self._metadata["mapping_details"].append({
                "node_name": node_name,
                "shape": (rows, cols, batch_size),
                "latency": mapping_result.latency,
                "utilization": mapping_result.get_compute_utilization()
            })

            modified = True

        return modified
