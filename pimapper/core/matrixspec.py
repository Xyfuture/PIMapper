from __future__ import annotations

"""Core domain model for MatrixMachine simulations.

The module is organised around three layers:

1. Matrix primitives (`MatrixShape`, `Tile`) describe how data is partitioned.
2. Hardware specifications (`PIMChannelSpec`, `AcceleratorSpec`) capture immutable
   configuration that can be instantiated into runtime objects.
3. Runtime entities (`PIMChannel`, `Accelerator`, `Mapping`) encapsulate mutable state
   such as live channel metadata or tile ownership.

Keeping the configuration separate from instances makes it easier to reason
about how an accelerator should look (spec) versus how it currently behaves (runtime
objects). Builders provided on the runtime classes help translate specs into
instantiated objects with minimal boilerplate.
"""

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .hwspec import Accelerator, PIMChannel


# ---------------------------------------------------------------------------
# Matrix primitives
# ---------------------------------------------------------------------------


class DataType(Enum):
    """Supported data types for matrix computations."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"

    @property
    def bytes_per_element(self) -> float:
        """Return the number of bytes per element for this data type."""
        return {
            DataType.FP16: 2.0,
            DataType.INT8: 1.0,
            DataType.INT4: 0.5,  # 4 bits = 0.5 bytes
            DataType.FP8: 1.0,
        }[self]


@dataclass(frozen=True)
class DataFormat:
    """Data format configuration for matrix computations."""

    input_dtype: DataType = DataType.FP16
    output_dtype: DataType = DataType.FP16
    weight_dtype: DataType = DataType.FP16

    def total_bytes_per_element(self) -> float:
        """Return the total bytes per element considering all data types.

        For a matrix multiplication, this represents the combined data footprint
        of input, weight, and output for a single computation.
        """
        return (
            self.input_dtype.bytes_per_element +
            self.output_dtype.bytes_per_element +
            self.weight_dtype.bytes_per_element
        )

    def __str__(self) -> str:
        return f"DataFormat(input={self.input_dtype.value}, output={self.output_dtype.value}, weight={self.weight_dtype.value})"


@dataclass(frozen=True)
class MatrixShape:
    """Matrix shape with convenience helpers for GEMV-like operations."""

    rows: int
    cols: int
    batch_size: int = 1
    data_format: DataFormat = DataFormat()

    @property
    def area(self) -> int:
        """Total number of elements in the matrix."""
        return self.rows * self.cols

    @property
    def volume(self) -> int:
        """Total number of elements including batch dimension."""
        return self.rows * self.cols * self.batch_size

    def to_tuple(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def contains(self, row: int, col: int, batch: int = 0) -> bool:
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                0 <= batch < self.batch_size)


    def __str__(self) -> str:
        if self.batch_size == 1:
            return f"MatrixShape({self.rows}×{self.cols}, {self.data_format})"
        else:
            return f"MatrixShape({self.rows}×{self.cols}×{self.batch_size}, {self.data_format})"


@dataclass(frozen=True)
class Tile:
    """Tile representing a submatrix with dimensions only (no position information)."""

    tile_id: str
    num_rows: int
    num_cols: int
    num_batches: int = 1
    data_format: DataFormat = DataFormat()

    _id_counter: ClassVar[int] = 1

    @classmethod
    def create(
        cls,
        num_rows: int,
        num_cols: int,
        num_batches: int = 1,
        *,
        prefix: str = "tile",
        data_format: Optional["DataFormat"] = None,
    ) -> "Tile":
        """Create a new tile with an auto-generated identifier."""

        tile_id = f"{prefix}_{cls._id_counter}"
        cls._id_counter += 1
        return cls(
            tile_id=tile_id,
            num_rows=num_rows,
            num_cols=num_cols,
            num_batches=num_batches,
            data_format=data_format or DataFormat()
        )

  
    @property
    def rows(self) -> int:
        return self.num_rows

    @property
    def cols(self) -> int:
        return self.num_cols

    @property
    def batches(self) -> int:
        return self.num_batches

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_rows, self.num_cols
    @property
    def area(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def volume(self) -> int:
        return self.num_rows * self.num_cols * self.num_batches

    def __str__(self) -> str:
        if self.num_batches == 1:
            return f"Tile({self.tile_id}:{self.num_rows}×{self.num_cols})"
        else:
            return f"Tile({self.tile_id}:{self.num_rows}×{self.num_cols}×{self.num_batches})"



# ---------------------------------------------------------------------------
# Mapping and validation
# ---------------------------------------------------------------------------

TileAssignmentInput = Tuple[str, int, int, int]  # (channel_id, num_rows, num_cols, num_batches)


@dataclass
class Mapping:
    """Bidirectional mapping between tiles and PIM channels."""

    matrix: MatrixShape
    accelerator: "Accelerator"
    tiles: Dict[str, Tile] = field(default_factory=dict)
    placement: Dict[str, List[Tile]] = field(default_factory=dict)
    reverse_placement: Dict[str, "PIMChannel"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for channel_id in self.accelerator.channels:
            self.placement.setdefault(channel_id, [])

    # Backwards compatibility property
    @property
    def chip(self) -> "Accelerator":
        """Backwards compatibility: alias for accelerator."""
        return self.accelerator

    @classmethod
    def from_tile_data(
        cls,
        matrix: MatrixShape,
        accelerator: "Accelerator",
        tile_data: Iterable[TileAssignmentInput],
    ) -> "Mapping":
        mapping = cls(matrix=matrix, accelerator=accelerator)
        mapping.build(tile_data)
        mapping.check_all()
        return mapping

    def add_tile(
        self,
        channel_id: str,
        num_rows: int,
        num_cols: int,
        num_batches: int = 1,
        *,
        tile_id: Optional[str] = None,
    ) -> Tile:
        tile = (
            Tile(
                tile_id=tile_id,
                num_rows=num_rows,
                num_cols=num_cols,
                num_batches=num_batches,
                data_format=self.matrix.data_format
            )
            if tile_id is not None
            else Tile.create(
                num_rows=num_rows,
                num_cols=num_cols,
                num_batches=num_batches,
                data_format=self.matrix.data_format
            )
        )
        self._register_tile(channel_id, tile)
        return tile

    def build(self, tile_data: Iterable[TileAssignmentInput]) -> None:
        for channel_id, num_rows, num_cols, num_batches in tile_data:
            self.add_tile(channel_id, num_rows, num_cols, num_batches)

    def _register_tile(self, channel_id: str, tile: Tile) -> None:
        if channel_id not in self.accelerator.channels:
            raise ValueError(f"Unknown PIM channel: {channel_id}")

        self.tiles[tile.tile_id] = tile
        self.placement.setdefault(channel_id, []).append(tile)
        self.reverse_placement[tile.tile_id] = self.accelerator.channels[channel_id]

    # ----------
    # Validation helpers
    # ----------

    def check_all(self) -> None:
        """Execute full validation; raises ValueError on failure."""
        self._check_tiles_exist_in_placement()
        self._check_full_coverage()

    def _check_tiles_exist_in_placement(self) -> None:
        seen = set()
        for channel, tiles in self.placement.items():
            for tile in tiles:
                if tile.tile_id not in self.tiles:
                    raise ValueError(
                        f"Placement referenced non-existent tile: {tile.tile_id} @ channel={channel}"
                    )
                if tile.tile_id in seen:
                    raise ValueError(f"Tile appears on multiple channels: {tile.tile_id}")
                seen.add(tile.tile_id)
        if seen != set(self.tiles.keys()):
            missing = set(self.tiles.keys()) - seen
            raise ValueError(f"Unassigned tiles exist: {sorted(missing)}")

    def _check_full_coverage(self) -> None:
        total = sum(t.volume for t in self.tiles.values())
        if total != self.matrix.volume:
            raise ValueError(
                "Coverage volume not equal to matrix volume: "
                f"tiles={total}, matrix={self.matrix.volume}"
            )

    # ----------
    # Convenience statistics
    # ----------

    def channel_loads(self) -> Dict[str, int]:
        return {channel: len(tiles) for channel, tiles in self.placement.items()}

    def channel_areas(self) -> Dict[str, int]:
        return {
            channel: sum(tile.area for tile in tiles)
            for channel, tiles in self.placement.items()
        }

    def channel_volumes(self) -> Dict[str, int]:
        return {
            channel: sum(tile.volume for tile in tiles)
            for channel, tiles in self.placement.items()
        }

    def tiles_of_channel(self, channel_id: str) -> List[Tile]:
        return list(self.placement.get(channel_id, []))

    # Backwards compatibility methods
    def die_loads(self) -> Dict[str, int]:
        return self.channel_loads()

    def die_areas(self) -> Dict[str, int]:
        return self.channel_areas()

    def die_volumes(self) -> Dict[str, int]:
        return self.channel_volumes()

    def tiles_of_die(self, die_id: str) -> List[Tile]:
        return self.tiles_of_channel(die_id)

    def display(self) -> None:
        """Display mapping summary with matrix size and tile assignments per channel."""
        print(f"Matrix: {self.matrix.rows}×{self.matrix.cols}×{self.matrix.batch_size}")
        for channel_id in sorted(self.placement.keys()):
            tiles = self.placement[channel_id]
            if tiles:
                tile_info = []
                for tile in tiles:
                    tile_info.append(f"{tile.num_rows}×{tile.num_cols}×{tile.num_batches}")
                print(f"  {channel_id}: {', '.join(tile_info)}")
            else:
                print(f"  {channel_id}: (empty)")

    # ----------
    # Bidirectional mapping checks
    # ----------

    def check_bidirectional_mapping(self) -> bool:
        for channel_id, tiles in self.placement.items():
            for tile in tiles:
                pim_channel = self.reverse_placement.get(tile.tile_id)
                if pim_channel is None or pim_channel.channel_id != channel_id:
                    return False

        for tile_id, pim_channel in self.reverse_placement.items():
            if tile_id not in self.tiles:
                return False
            assigned_tiles = self.placement.get(pim_channel.channel_id, [])
            if not any(tile.tile_id == tile_id for tile in assigned_tiles):
                return False

        for channel_id, tiles in self.placement.items():
            for tile in tiles:
                if tile.tile_id not in self.reverse_placement:
                    return False

        return True

    def __str__(self) -> str:
        tile_count = len(self.tiles)
        channel_count = len([channel for channel, tiles in self.placement.items() if tiles])
        return f"Mapping({tile_count} tiles on {channel_count} active channels, matrix={self.matrix})"
