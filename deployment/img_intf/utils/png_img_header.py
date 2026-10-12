"""
png_img_header.py

Lightweight PNG file introspection used only for local logging/validation on
the client side (e.g. reporting width/height before a transfer). This does
NOT participate in the wire protocol — the server treats PNG files as an
opaque byte blob (see generic_img_header.py).
"""

import struct
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IHDR_CHUNK_TYPE = b"IHDR"

# PNG signature (8) + IHDR chunk: length(4) + type(4) + data(13) + crc(4) = 25
IHDR_DATA_OFFSET = len(PNG_SIGNATURE) + 8
HEADER_SIZE_BYTES = IHDR_DATA_OFFSET + 13 + 4  # 33


@dataclass(frozen=True)
class PNGHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression_method: int
    filter_method: int
    interlace_method: int


def parse_png_header_from_file(file_path: str) -> PNGHeader:
    with open(file_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE_BYTES)
    return parse_png_header(header_bytes)


def parse_png_header(header_bytes: bytes) -> PNGHeader:
    if len(header_bytes) < HEADER_SIZE_BYTES:
        raise ValueError(
            f"Need at least {HEADER_SIZE_BYTES} bytes for a PNG header, got {len(header_bytes)}"
        )

    if header_bytes[:8] != PNG_SIGNATURE:
        raise ValueError("Not a PNG file (bad signature)")

    length, chunk_type = struct.unpack("!I4s", header_bytes[8:16])
    if chunk_type != IHDR_CHUNK_TYPE or length != 13:
        raise ValueError(f"Expected a 13-byte IHDR chunk first, got type={chunk_type!r} length={length}")

    (width, height, bit_depth, color_type, compression_method,
     filter_method, interlace_method) = struct.unpack("!IIBBBBB", header_bytes[16:29])

    return PNGHeader(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        compression_method=compression_method,
        filter_method=filter_method,
        interlace_method=interlace_method,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r".\input\sample.png"
    header = parse_png_header_from_file(path)
    print(f"Width: {header.width}, Height: {header.height}, Bit depth: {header.bit_depth}, Color type: {header.color_type}")
