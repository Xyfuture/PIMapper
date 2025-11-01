from dataclasses import dataclass
from .matrixspec import Mapping


@dataclass
class MappingResult:
    mapping: Mapping
    latency: float

    def get_chip_total_compute_power_gops(self) -> float:
        return self.mapping.chip.total_compute_power_gops

    def get_matrix_operation_count(self) -> int:
        return self.mapping.matrix.area * self.mapping.matrix.batch_size

    def get_compute_utilization(self) -> float:
        matrix_operation_count = self.get_matrix_operation_count()
        total_compute_power = self.get_chip_total_compute_power_gops()
        total_available_compute = total_compute_power * self.latency
        return matrix_operation_count / total_available_compute if total_available_compute > 0 else 0.0


def calculate_compute_utilization(mapping_result: MappingResult) -> float:
    return mapping_result.get_compute_utilization()
