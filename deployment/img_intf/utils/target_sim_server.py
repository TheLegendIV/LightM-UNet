
import asyncio
import struct
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.batch_img_header import parse_batch_header, MARKER_IMAGE, MARKER_STOP

"""
Local software stand-in for the LwIP server on the board (see
deployment/FINN_interface/lwip_v1/src/run.c). Speaks the same v3 batch
marker protocol as BatchImageStreamer/pack_and_send.py (IMG3 header, 1-byte
MARKER_IMAGE/MARKER_STOP per unit, ACK framing) and echoes each image's
payload straight back (loopback-style -- matches the server's
ACCEL_LOOPBACK=1 behavior), so input/output sizes must be equal. Useful for
exercising the PC-side client without needing real hardware.
"""


def _render_progress_bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        filled = 0
    else:
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(ratio * width)
    bar = "=" * filled + "-" * (width - filled)
    return f"0%  [{bar}]  100%"


def _write_progress(prefix: str, done: int, total: int, *, width: int = 40):
    line = f"{prefix} {_render_progress_bar(done, total, width=width)}"
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if total > 0 and done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


async def _wait_closed(writer):
    # Older asyncio StreamWriter implementations don't expose wait_closed().
    if hasattr(writer, "wait_closed"):
        await writer.wait_closed()


async def handle_client(reader, writer):
    print("[SERVER] Client connected")

    # Batch header sent once per session: [4-byte BE length][16-byte IMG3 header].
    header_len_bytes = await reader.readexactly(4)
    header_len = struct.unpack("!I", header_len_bytes)[0]
    header = await reader.readexactly(header_len)
    batch = parse_batch_header(header)
    payload_length = int(batch.payload_length)
    print(
        f"[SERVER] Received IMG3 header: payload_length={payload_length} "
        f"({batch.width}x{batch.height})"
    )

    # Header ACK: "ACK" + expected_output_bytes (u32 BE). Loopback mode ->
    # output size equals input size.
    writer.write(b"ACK" + struct.pack("!I", payload_length))
    await writer.drain()
    print("[SERVER] Sent header ACK")

    num_images = 0
    while True:
        marker = (await reader.readexactly(1))[0]

        if marker == MARKER_STOP:
            writer.write(b"ACK")
            await writer.drain()
            print(f"[SERVER] Received STOP marker after {num_images} image(s); sent ACK")
            break

        if marker != MARKER_IMAGE:
            print(f"[SERVER] ERROR: unexpected marker byte 0x{marker:02x}")
            break

        pixels = await reader.readexactly(payload_length)
        num_images += 1
        _write_progress("[SERVER]", num_images, num_images)

        # Loopback: echo the payload straight back (mirrors ACCEL_LOOPBACK=1
        # on the real server), then per-image ACK.
        writer.write(pixels)
        await writer.drain()
        writer.write(b"ACK")
        await writer.drain()
        print(f"[SERVER] Echoed image {num_images} ({len(pixels)} bytes) and sent ACK")

    writer.close()
    await _wait_closed(writer)
    print("[SERVER] Connection closed")

def main(host: str = "127.0.0.1", port: int = 5001):
    loop = asyncio.get_event_loop()
    server = loop.run_until_complete(asyncio.start_server(handle_client, host, port))
    print(f"[SERVER] Listening on {host}:{port}")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()


if __name__ == "__main__":
    main()