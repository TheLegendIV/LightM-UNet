import os
import socket
import struct
import sys
import tempfile
import threading
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.send_image import (  # noqa: E402
    BatchImageStreamer, ACCEL_INPUT_BYTES, ACCEL_OUTPUT_BYTES,
)
from utils.batch_img_header import MAGIC, HEADER_SIZE_BYTES, MARKER_STOP  # noqa: E402

NUM_IMAGES = 10


class MockLoopbackServer:
    """Python stand-in for run.c's ACCEL_LOOPBACK v3 marker protocol: echoes
    each image back zero-padded to ACCEL_OUTPUT_BYTES, per-image ACK after
    every image, stop-ack + close on MARKER_STOP."""

    def __init__(self, payload_length: int, output_bytes: int):
        self.payload_length = payload_length
        self.output_bytes = output_bytes
        self.images_received = 0
        self.error = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self._sock.settimeout(10)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    @staticmethod
    def _recv_exact(conn, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed connection early")
            buf.extend(chunk)
        return bytes(buf)

    def _run(self):
        try:
            conn, _ = self._sock.accept()
            with conn:
                struct.unpack("!I", self._recv_exact(conn, 4))  # length prefix, ignored
                header = self._recv_exact(conn, HEADER_SIZE_BYTES)
                magic, payload_length = struct.unpack("!4sI", header[:8])
                if magic != MAGIC or payload_length != self.payload_length:
                    raise ValueError(
                        f"unexpected header: magic={magic!r} payload_length={payload_length}"
                    )

                conn.sendall(b"ACK" + struct.pack("!I", self.output_bytes))

                while True:
                    marker = self._recv_exact(conn, 1)[0]
                    if marker == MARKER_STOP:
                        conn.sendall(b"ACK")
                        break
                    data = self._recv_exact(conn, self.payload_length)
                    copy_len = min(self.payload_length, self.output_bytes)
                    result = data[:copy_len] + bytes(self.output_bytes - copy_len)
                    conn.sendall(result)
                    conn.sendall(b"ACK")
                    self.images_received += 1
        except Exception as exc:  # noqa: BLE001 - surfaced via check() on the test thread
            self.error = exc
        finally:
            self._sock.close()

    def check(self):
        if self.error is not None:
            raise self.error


class TestBatchMarkerProtocol(unittest.TestCase):
    def test_ten_images_roundtrip_bit_exact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)

            # Each image's pixels are all set to (index + 1), so its value
            # denotes its position in the sequence.
            sent_data = {}
            for i in range(NUM_IMAGES):
                value = i + 1
                data = bytes([value]) * ACCEL_INPUT_BYTES
                name = f"seq_{i:02d}.raw"
                with open(os.path.join(input_dir, name), "wb") as f:
                    f.write(data)
                sent_data[name] = (i, data)

            server = MockLoopbackServer(ACCEL_INPUT_BYTES, ACCEL_OUTPUT_BYTES)
            server.start()

            streamer = BatchImageStreamer(
                "127.0.0.1", server.port, input_dir, output_dir,
                pattern="seq_*.raw", verify_loopback=False,
            )
            all_ok = streamer.run()
            server.join(timeout=10)
            server.check()

            self.assertTrue(all_ok)

            # Total images in == total images out.
            output_files = sorted(os.listdir(output_dir))
            self.assertEqual(len(sent_data), NUM_IMAGES)
            self.assertEqual(len(output_files), NUM_IMAGES)
            self.assertEqual(server.images_received, NUM_IMAGES)

            # Bit-by-bit check for each image, in sequence order.
            for name, (index, original) in sent_data.items():
                out_path = os.path.join(output_dir, name)
                self.assertTrue(os.path.exists(out_path), f"missing output for {name}")
                with open(out_path, "rb") as f:
                    result = f.read()

                self.assertEqual(len(result), ACCEL_OUTPUT_BYTES)
                pixels, padding = result[:ACCEL_INPUT_BYTES], result[ACCEL_INPUT_BYTES:]

                for byte_offset, (expected, actual) in enumerate(zip(original, pixels)):
                    self.assertEqual(
                        actual, expected,
                        f"{name}: byte {byte_offset} mismatch (expected {expected}, got {actual})"
                    )
                self.assertEqual(pixels, original, f"{name}: pixel region does not match sent image")
                self.assertTrue(all(b == 0 for b in padding), f"{name}: zero-padding region not all zero")

                expected_value = index + 1
                self.assertTrue(
                    all(b == expected_value for b in pixels),
                    f"{name}: expected all pixels == {expected_value} (sequence position)"
                )


if __name__ == "__main__":
    unittest.main()
