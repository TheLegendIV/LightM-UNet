"""
batch_img_header.py

Wire-level header for the per-image-marker image transfer protocol used
between the img_intf Python client and the LwIP loopback
server (deployment/FINN_interface/test/src/run.c).

This header is sent exactly ONCE per connection/session, up front. It tells
the server the fixed per-image payload size (every image that follows must
be exactly this many bytes -- the accelerator has a fixed input tensor
shape). The client owns the image count entirely: it precedes every image
with a 1-byte marker (see marker constants below) and a final STOP marker
when done, so num_images is not part of the header.

Wire layout (16 bytes, all fields network byte order / big-endian):
    magic           4s   b"IMG3"
    payload_length  I    fixed per-image payload size in bytes (e.g. 4096)
    width           I    informational only, 0 if unknown/not applicable
    height          I    informational only, 0 if unknown/not applicable

Per-image marker byte (sent by the client before each unit that follows the
header ACK):
    MARKER_STOP  (0x00): no payload follows; ends the session.
    MARKER_IMAGE (0x01): exactly payload_length raw bytes follow.
"""

import struct
from dataclasses import dataclass

MAGIC = b"IMG3"
# Pipelined streaming protocol (deployment/FINN_interface/lwip_stream): same
# 16-byte layout, different magic so a lockstep client and a streaming server
# (or vice versa) fail loudly instead of deadlocking.
MAGIC_STREAM = b"IMGS"
# Test-split benchmark server (deployment/FINN_interface/lwip_collect_results): same layout, own magic.
MAGIC_COLLECT = b"IMGC"
HEADER_SIZE_BYTES = 16
_STRUCT_FMT = "!4sIII"

MARKER_STOP = 0x00
MARKER_IMAGE = 0x01
# lwip_collect_results commands (see img_intf/interfaces/collect_split.py and the server's run.c):
#   MARKER_LOAD    + payload          : upload one image into board DDR (no reply)
#   MARKER_LATENCY                    : lockstep per image; reply "LAT" + u32 n + n x 3 u32 PL ticks
#   MARKER_RUN     + u8 mode + u8 reps: run the whole split `reps` times; reply "RUN" + u32 reps + reps x 3 u32 PL ticks
#   MARKER_FETCH                      : reply "OUT" + u32 n, then n class maps
#   MARKER_POWER   + u8 mode + u8 warmup_s + u8 window_s : sample the board's INA226 rails (mode 0: accelerator idle,
#                     mode 1: the loaded split runs back-to-back); reply "PWR" + u32 18 + raw per-device sums (collect_split.py)
MARKER_LOAD = 0x03
MARKER_LATENCY = 0x04
MARKER_RUN = 0x05
MARKER_FETCH = 0x06
MARKER_POWER = 0x07


@dataclass(frozen=True)
class BatchHeader:
    magic: bytes
    payload_length: int
    width: int
    height: int


def build_batch_header(payload_length: int, width: int = 0, height: int = 0,
                       magic: bytes = MAGIC) -> bytes:
    return struct.pack(_STRUCT_FMT, magic, payload_length, width, height)


def parse_batch_header(header_bytes: bytes, magic: bytes = MAGIC) -> BatchHeader:
    if len(header_bytes) < HEADER_SIZE_BYTES:
        raise ValueError(
            f"Need at least {HEADER_SIZE_BYTES} bytes for batch image header, got {len(header_bytes)}"
        )

    got_magic, payload_length, width, height = struct.unpack(
        _STRUCT_FMT, header_bytes[:HEADER_SIZE_BYTES]
    )
    if got_magic != magic:
        raise ValueError(f"Not a batch image header (bad magic: {got_magic!r}, expected {magic!r})")

    return BatchHeader(magic=got_magic, payload_length=payload_length, width=width, height=height)
