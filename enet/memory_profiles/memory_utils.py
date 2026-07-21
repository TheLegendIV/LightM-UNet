from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
}


@dataclass(frozen=True)
class TensorRecord:
    path: str
    name: str
    shape: tuple[int, ...]
    note: str = ""

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def bytes(self, bytes_per_element: int) -> int:
        return self.elements * bytes_per_element


@dataclass(frozen=True)
class WorkingRecord:
    path: str
    module: str
    reason: str
    tensors: tuple[TensorRecord, ...]
    note: str = ""

    def bytes(self, bytes_per_element: int) -> int:
        return sum(tensor.bytes(bytes_per_element) for tensor in self.tensors)


@dataclass(frozen=True)
class ParameterRecord:
    path: str
    name: str
    shape: tuple[int, ...]
    elements: int
    note: str = ""

    def bytes(self, bytes_per_element: int) -> int:
        return self.elements * bytes_per_element


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) != 6:
        raise ValueError("--channels must provide six comma-separated integers.")
    if any(channel <= 0 for channel in channels):
        raise ValueError("--channels must all be positive.")
    return channels


def dtype_bytes(dtype: str) -> int:
    key = dtype.lower()
    if key not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {supported}")
    return DTYPE_BYTES[key]


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def shape_str(shape: Iterable[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def downsampled_hw(height: int, width: int, stage_index: int) -> tuple[int, int]:
    divisor = 2 ** stage_index
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Height and width must be divisible by {divisor} for stage {stage_index + 1}."
        )
    return height // divisor, width // divisor


def encoder_shapes(
    batch: int,
    height: int,
    width: int,
    channels: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    shapes = []
    for stage_index, channel in enumerate(channels):
        h, w = downsampled_hw(height, width, stage_index)
        shapes.append((batch, channel, h, w))
    return shapes


def print_tensor_table(title: str, records: list[TensorRecord], bytes_per_element: int) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Path':<10} {'Name':<32} {'Shape':<24} {'MiB':>10}  Note")
    print("-" * 96)
    for record in records:
        print(
            f"{record.path:<10} "
            f"{record.name:<32} "
            f"{shape_str(record.shape):<24} "
            f"{mib(record.bytes(bytes_per_element)):>10.3f}  "
            f"{record.note}"
        )


def print_path_summary(records: list[TensorRecord], bytes_per_element: int) -> None:
    totals: dict[str, int] = {}
    for record in records:
        totals[record.path] = totals.get(record.path, 0) + record.bytes(bytes_per_element)

    print("\n=== Grouped Totals ===")
    print(f"{'Path':<10} {'MiB':>10}")
    print("-" * 24)
    for path, total in sorted(totals.items()):
        print(f"{path:<10} {mib(total):>10.3f}")
    print("-" * 24)
    print(f"{'total':<10} {mib(sum(totals.values())):>10.3f}")


def print_parameter_table(title: str, records: list[ParameterRecord], bytes_per_element: int) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Path':<10} {'Name':<56} {'Shape':<20} {'Params':>12} {'MiB':>10}")
    print("-" * 116)
    for record in records:
        print(
            f"{record.path:<10} "
            f"{record.name:<56} "
            f"{shape_str(record.shape):<20} "
            f"{record.elements:>12,} "
            f"{mib(record.bytes(bytes_per_element)):>10.3f}"
        )


def print_parameter_summary(records: list[ParameterRecord], bytes_per_element: int) -> None:
    totals: dict[str, int] = {}
    for record in records:
        totals[record.path] = totals.get(record.path, 0) + record.bytes(bytes_per_element)

    print("\n=== Parameter Grouped Totals ===")
    print(f"{'Path':<10} {'Params':>12} {'MiB':>10}")
    print("-" * 38)
    for path in sorted(totals):
        params = sum(record.elements for record in records if record.path == path)
        print(f"{path:<10} {params:>12,} {mib(totals[path]):>10.3f}")
    print("-" * 38)
    print(
        f"{'total':<10} "
        f"{sum(record.elements for record in records):>12,} "
        f"{mib(sum(totals.values())):>10.3f}"
    )


def print_working_table(title: str, records: list[WorkingRecord], bytes_per_element: int) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Path':<10} {'Module':<34} {'Reason':<22} {'MiB':>10}  Tensors")
    print("-" * 112)
    for record in records:
        tensors = " + ".join(f"{tensor.name}{shape_str(tensor.shape)}" for tensor in record.tensors)
        note = f" ({record.note})" if record.note else ""
        print(
            f"{record.path:<10} "
            f"{record.module:<34} "
            f"{record.reason:<22} "
            f"{mib(record.bytes(bytes_per_element)):>10.3f}  "
            f"{tensors}{note}"
        )


def print_working_summary(records: list[WorkingRecord], bytes_per_element: int) -> None:
    if not records:
        return

    totals: dict[str, int] = {}
    for record in records:
        totals[record.path] = totals.get(record.path, 0) + record.bytes(bytes_per_element)

    peak = max(records, key=lambda record: record.bytes(bytes_per_element))
    print("\n=== Local Working Summary ===")
    print(f"{'Path':<10} {'Sum Listed Peaks MiB':>24}")
    print("-" * 40)
    for path, total in sorted(totals.items()):
        print(f"{path:<10} {mib(total):>24.3f}")
    print("-" * 40)
    print(
        "Largest local working set: "
        f"{peak.module} ({peak.reason}) = {mib(peak.bytes(bytes_per_element)):.3f} MiB"
    )
    print("Listed peaks are not assumed to be simultaneous.")
