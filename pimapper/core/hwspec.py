from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Hardware specification layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeDieSpec:
    """Immutable specification for a compute die.

    Args:
        compute_power: Compute power in TFLOPS
        input_bandwidth: Input bandwidth in GB/s (optional if shared_bandwidth is set)
        output_bandwidth: Output bandwidth in GB/s (optional if shared_bandwidth is set)
        memory_bandwidth: Memory bandwidth in TB/s
        shared_bandwidth: Shared I/O bandwidth in GB/s (mutually exclusive with input/output bandwidth)
    """

    compute_power: float  # TFLOPS
    input_bandwidth: Optional[float] = None  # GB/s
    output_bandwidth: Optional[float] = None  # GB/s
    memory_bandwidth: float = 0.0  # TB/s
    shared_bandwidth: Optional[float] = None  # GB/s

    def __post_init__(self) -> None:
        if self.compute_power <= 0:
            raise ValueError("compute_power must be positive")
        if self.memory_bandwidth <= 0:
            raise ValueError("memory_bandwidth must be positive")

        # Validate bandwidth configuration
        has_separate = self.input_bandwidth is not None or self.output_bandwidth is not None
        has_shared = self.shared_bandwidth is not None

        if has_separate and has_shared:
            raise ValueError(
                "Cannot specify both separate bandwidths (input_bandwidth/output_bandwidth) "
                "and shared_bandwidth at the same time"
            )

        if not has_separate and not has_shared:
            raise ValueError(
                "Must specify either separate bandwidths (input_bandwidth and output_bandwidth) "
                "or shared_bandwidth"
            )

        if has_separate:
            if self.input_bandwidth is None or self.input_bandwidth <= 0:
                raise ValueError("input_bandwidth must be positive when using separate bandwidths")
            if self.output_bandwidth is None or self.output_bandwidth <= 0:
                raise ValueError("output_bandwidth must be positive when using separate bandwidths")

        if has_shared:
            if self.shared_bandwidth is None or self.shared_bandwidth <= 0:
                raise ValueError("shared_bandwidth must be positive when using shared bandwidth mode")

    def get_input_bandwidth(self) -> float:
        """Get the effective input bandwidth.

        Returns shared_bandwidth if in shared mode, otherwise returns input_bandwidth.
        """
        if self.shared_bandwidth is not None:
            return self.shared_bandwidth
        assert self.input_bandwidth is not None
        return self.input_bandwidth

    def get_output_bandwidth(self) -> float:
        """Get the effective output bandwidth.

        Returns shared_bandwidth if in shared mode, otherwise returns output_bandwidth.
        """
        if self.shared_bandwidth is not None:
            return self.shared_bandwidth
        assert self.output_bandwidth is not None
        return self.output_bandwidth

    def __str__(self) -> str:
        if self.shared_bandwidth is not None:
            return (f"ComputeDieSpec(compute={self.compute_power}TFLOPS, "
                    f"shared_bandwidth={self.shared_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")
        else:
            return (f"ComputeDieSpec(compute={self.compute_power}TFLOPS, "
                    f"input={self.input_bandwidth}GB/s, output={self.output_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")


@dataclass(frozen=True)
class ChipSpec:
    """Immutable blueprint for a chip composed of homogeneous dies."""

    die_count: int
    die_spec: ComputeDieSpec

    def __post_init__(self) -> None:
        if self.die_count <= 0:
            raise ValueError("die_count must be positive")

    def __str__(self) -> str:
        return f"ChipSpec({self.die_count} dies, {self.die_spec})"

# Backwards compatibility aliases (prefer the *Spec* names going forward).
ComputeDieConfig = ComputeDieSpec
ChipConfig = ChipSpec


# ---------------------------------------------------------------------------
# Runtime hardware entities
# ---------------------------------------------------------------------------


@dataclass
class ComputeDie:
    """Runtime representation of a compute die."""

    die_id: str
    spec: ComputeDieSpec
    meta: Dict[str, str] = field(default_factory=dict)

  
    @property
    def compute_power(self) -> float:
        return self.spec.compute_power

    @property
    def input_bandwidth(self) -> float:
        return self.spec.get_input_bandwidth()

    @property
    def output_bandwidth(self) -> float:
        return self.spec.get_output_bandwidth()

    @property
    def memory_bandwidth(self) -> float:
        return self.spec.memory_bandwidth

    def __str__(self) -> str:
        meta_str = f", meta={self.meta}" if self.meta else ""
        return f"ComputeDie({self.die_id}, {self.spec}{meta_str})"


@dataclass
class Chip:
    """Runtime chip composed of instantiated compute dies."""

    spec: ChipSpec
    compute_dies: Dict[str, ComputeDie] = field(default_factory=dict)

    @classmethod
    def create_from_spec(
        cls,
        spec: ChipSpec,
        *,
        id_prefix: str = "die",
        meta_factory: Optional[Callable[[int], Dict[str, str]]] = None,
    ) -> "Chip":
        """Create a Chip instance populated with compute dies from the spec."""

        compute_dies: Dict[str, ComputeDie] = {}
        for idx in range(spec.die_count):
            die_id = f"{id_prefix}_{idx}"
            meta = meta_factory(idx) if meta_factory else {}
            compute_dies[die_id] = ComputeDie(die_id=die_id, spec=spec.die_spec, meta=meta)
        return cls(spec=spec, compute_dies=compute_dies)

    def add_die(self, die: ComputeDie) -> None:
        self.compute_dies[die.die_id] = die

    def remove_die(self, die_id: str) -> None:
        self.compute_dies.pop(die_id, None)

    def get_die(self, die_id: str) -> Optional[ComputeDie]:
        return self.compute_dies.get(die_id)

    @property
    def total_compute_power(self) -> float:
        return sum(die.compute_power for die in self.compute_dies.values())

    @property
    def total_compute_power_gops(self) -> float:
        return self.total_compute_power * 10**3  # Convert TOPS to GOPS

    @property
    def total_input_bandwidth(self) -> float:
        return sum(die.input_bandwidth for die in self.compute_dies.values())

    @property
    def total_output_bandwidth(self) -> float:
        return sum(die.output_bandwidth for die in self.compute_dies.values())

    @property
    def total_bandwidth(self) -> float:
        return self.total_input_bandwidth + self.total_output_bandwidth

    @property
    def total_memory_bandwidth(self) -> float:
        return sum(die.memory_bandwidth for die in self.compute_dies.values())

    def __str__(self) -> str:
        die_count = len(self.compute_dies)
        total_compute = self.total_compute_power
        return f"Chip({die_count} dies, {total_compute}TFLOPS total compute)"
