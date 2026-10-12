"""
generic_img_header.py

Wire-level header used between the img_intf Python client and
the (now format-agnostic) LwIP echo/loopback server. Unlike the BMP/FXD
per-format headers, this header carries no format-specific fields — it just
tells the server how many opaque payload bytes follow, plus optional
width/height for logging. This lets the server echo back any file type
(BMP, FXD, PNG, ...) without understanding its internal layout.

Wire layout (16 bytes, all fields network byte order / big-endian):
    magic           4s   b"IMG1"
    payload_length  I    number of raw bytes that follow (the whole input file)
    width           I    informational only, 0 if unknown/not applicable
    height          I    informational only, 0 if unknown/not applicable
"""

import struct
from dataclasses import dataclass

MAGIC = b"IMG1"
HEADER_SIZE_BYTES = 16
_STRUCT_FMT = "!4sIII"


@dataclass(frozen=True)
class GenericHeader:
    magic: bytes
    payload_length: int
    width: int
    height: int


def build_generic_header(payload_length: int, width: int = 0, height: int = 0) -> bytes:
    return struct.pack(_STRUCT_FMT, MAGIC, payload_length, width, height)


def parse_generic_header(header_bytes: bytes) -> GenericHeader:
    if len(header_bytes) < HEADER_SIZE_BYTES:
        raise ValueError(
            f"Need at least {HEADER_SIZE_BYTES} bytes for generic image header, got {len(header_bytes)}"
        )

    magic, payload_length, width, height = struct.unpack(_STRUCT_FMT, header_bytes[:HEADER_SIZE_BYTES])
    if magic != MAGIC:
        raise ValueError(f"Not a generic image header (bad magic: {magic!r}, expected {MAGIC!r})")

    return GenericHeader(magic=magic, payload_length=payload_length, width=width, height=height)
