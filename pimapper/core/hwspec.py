from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Hardware specification layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostSpec:
    """Immutable specification for a host processor.

    Args:
        vector_compute_power: Vector compute power in GFLOPS
    """

    vector_compute_power: float  # GFLOPS

    def __post_init__(self) -> None:
        if self.vector_compute_power <= 0:
            raise ValueError("vector_compute_power must be positive")

    def __str__(self) -> str:
        return f"HostSpec(vector_compute={self.vector_compute_power}GFLOPS)"


@dataclass(frozen=True)
class PIMChannelSpec:
    """Immutable specification for a PIM channel.

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
            return (f"PIMChannelSpec(compute={self.compute_power}TFLOPS, "
                    f"shared_bandwidth={self.shared_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")
        else:
            return (f"PIMChannelSpec(compute={self.compute_power}TFLOPS, "
                    f"input={self.input_bandwidth}GB/s, output={self.output_bandwidth}GB/s, "
                    f"memory={self.memory_bandwidth}TB/s)")


@dataclass(frozen=True)
class AcceleratorSpec:
    """Immutable blueprint for an accelerator composed of homogeneous PIM channels."""

    channel_count: int
    channel_spec: PIMChannelSpec
    host_spec: Optional[HostSpec] = None

    def __post_init__(self) -> None:
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive")

    def __str__(self) -> str:
        return f"AcceleratorSpec({self.channel_count} channels, {self.channel_spec})"

# Backwards compatibility aliases
ComputeDieSpec = PIMChannelSpec
ComputeDieConfig = PIMChannelSpec
ChipSpec = AcceleratorSpec
ChipConfig = AcceleratorSpec


# ---------------------------------------------------------------------------
# Runtime hardware entities
# ---------------------------------------------------------------------------


@dataclass
class Host:
    """Runtime representation of a host processor that connects to PIM channels."""

    host_id: str
    spec: Optional[HostSpec] = None
    meta: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        spec_str = f", spec={self.spec}" if self.spec else ""
        meta_str = f", meta={self.meta}" if self.meta else ""
        return f"Host({self.host_id}{spec_str}{meta_str})"


@dataclass
class PIMChannel:
    """Runtime representation of a PIM channel."""

    channel_id: str
    spec: PIMChannelSpec
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
        return f"PIMChannel({self.channel_id}, {self.spec}{meta_str})"

# Backwards compatibility alias
ComputeDie = PIMChannel


@dataclass
class Accelerator:
    """Runtime accelerator composed of a host and PIM channels."""

    spec: AcceleratorSpec
    host: Optional[Host] = None
    channels: Dict[str, PIMChannel] = field(default_factory=dict)

    @classmethod
    def create_from_spec(
        cls,
        spec: AcceleratorSpec,
        *,
        id_prefix: str = "channel",
        host_id: str = "host_0",
        meta_factory: Optional[Callable[[int], Dict[str, str]]] = None,
    ) -> "Accelerator":
        """Create an Accelerator instance populated with PIM channels from the spec."""

        channels: Dict[str, PIMChannel] = {}
        for idx in range(spec.channel_count):
            channel_id = f"{id_prefix}_{idx}"
            meta = meta_factory(idx) if meta_factory else {}
            channels[channel_id] = PIMChannel(channel_id=channel_id, spec=spec.channel_spec, meta=meta)

        host = Host(host_id=host_id, spec=spec.host_spec)
        return cls(spec=spec, host=host, channels=channels)

    def add_channel(self, channel: PIMChannel) -> None:
        self.channels[channel.channel_id] = channel

    def remove_channel(self, channel_id: str) -> None:
        self.channels.pop(channel_id, None)

    def get_channel(self, channel_id: str) -> Optional[PIMChannel]:
        return self.channels.get(channel_id)

    @property
    def total_compute_power(self) -> float:
        return sum(channel.compute_power for channel in self.channels.values())

    @property
    def total_compute_power_gops(self) -> float:
        return self.total_compute_power * 10**3  # Convert TOPS to GOPS

    @property
    def total_input_bandwidth(self) -> float:
        return sum(channel.input_bandwidth for channel in self.channels.values())

    @property
    def total_output_bandwidth(self) -> float:
        return sum(channel.output_bandwidth for channel in self.channels.values())

    @property
    def total_bandwidth(self) -> float:
        return self.total_input_bandwidth + self.total_output_bandwidth

    @property
    def total_memory_bandwidth(self) -> float:
        return sum(channel.memory_bandwidth for channel in self.channels.values())

    def __str__(self) -> str:
        channel_count = len(self.channels)
        total_compute = self.total_compute_power
        return f"Accelerator({channel_count} channels, {total_compute}TFLOPS total compute)"

    # Backwards compatibility properties
    @property
    def compute_dies(self) -> Dict[str, PIMChannel]:
        """Backwards compatibility: alias for channels."""
        return self.channels

    def add_die(self, die: PIMChannel) -> None:
        """Backwards compatibility: alias for add_channel."""
        self.add_channel(die)

    def remove_die(self, die_id: str) -> None:
        """Backwards compatibility: alias for remove_channel."""
        self.remove_channel(die_id)

    def get_die(self, die_id: str) -> Optional[PIMChannel]:
        """Backwards compatibility: alias for get_channel."""
        return self.get_channel(die_id)

# Backwards compatibility alias
Chip = Accelerator
