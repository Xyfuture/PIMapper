"""Latency calculation pass using topological sorting.

This pass traverses the computation graph in topological order and calculates
the total latency by accumulating latencies from matrix operations and vector
operations that have been processed by MatrixMappingPass and VectorLatencyPass.
"""

import networkx as nx
from pimapper.modelmapper.passes.base import Pass
from pimapper.core.graph.base import NxComputationGraph


# Operation types that should have latency information
MATRIX_OPS = {"matmul", "fusion_matrix", "batched_matmul"}
VECTOR_OPS = {"vector_add", "vector_mul", "vector_dot", "silu", "softmax", "rmsnorm"}


class LatencyCalculationPass(Pass):
    """Pass that calculates total latency using topological sorting.

    This pass:
    1. Performs topological sort on the computation graph
    2. Traverses nodes in topological order
    3. Accumulates latency from operations that have latency information
    4. Records detailed latency information for each operation

    The pass expects that:
    - MatrixMappingPass has been run for matrix operations (matmul, fusion_matrix, batched_matmul)
    - VectorLatencyPass has been run for vector operations (vector_add, vector_mul, silu, etc.)

    Both passes store latency in op.kwargs['latency'].
    """

    def __init__(self, *, name: str = None):
        super().__init__(
            name=name or "LatencyCalculationPass",
            description="Calculates total latency using topological sorting"
        )

    def run(self, graph: NxComputationGraph) -> bool:
        """Execute the latency calculation pass.

        Args:
            graph: Computation graph to process

        Returns:
            False (does not modify the graph)
        """
        self._metadata = {
            "total_latency": 0.0,
            "matrix_ops_count": 0,
            "vector_ops_count": 0,
            "total_ops_count": 0,
            "latency_details": []
        }

        cumulative_latency = 0.0

        # Traverse in topological order
        for node_name in nx.topological_sort(graph.graph):
            op = graph.node_record(node_name)
            op_type = getattr(op, 'op_type', None)

            # Skip operations that don't have latency information
            if op_type not in (MATRIX_OPS | VECTOR_OPS):
                continue

            # Get latency from op.kwargs
            latency = None
            if hasattr(op, 'kwargs') and op.kwargs:
                latency = op.kwargs.get('latency')

            if latency is None:
                continue

            # Update cumulative latency
            cumulative_latency += latency

            # Determine operation category
            is_matrix_op = op_type in MATRIX_OPS
            is_vector_op = op_type in VECTOR_OPS

            # Extract shape information if available
            shape_info = {}
            if hasattr(op, 'results') and op.results:
                result = op.results[0]
                shape = getattr(result, 'shape', None)
                if shape:
                    shape_info["output_shape"] = shape

            # For matrix operations, also include mapping_result info if available
            if is_matrix_op and hasattr(op, 'kwargs'):
                mapping_result = op.kwargs.get('mapping_result')
                if mapping_result:
                    matrix = mapping_result.mapping.matrix
                    shape_info["matrix_shape"] = {
                        "rows": matrix.rows,
                        "cols": matrix.cols,
                        "batch_size": matrix.batch_size
                    }

            # Record details
            detail = {
                "node_name": node_name,
                "op_type": op_type,
                "category": "matrix" if is_matrix_op else "vector",
                "latency": latency,
                "cumulative_latency": cumulative_latency,
            }

            if shape_info:
                detail.update(shape_info)

            self._metadata["latency_details"].append(detail)

            # Update counts
            if is_matrix_op:
                self._metadata["matrix_ops_count"] += 1
            if is_vector_op:
                self._metadata["vector_ops_count"] += 1
            self._metadata["total_ops_count"] += 1

        self._metadata["total_latency"] = cumulative_latency

        return False
