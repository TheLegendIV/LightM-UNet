import asyncio

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[INFO] Connected to {addr}")
    try:
        while True:
            # Read length header
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, 'big')
            payload = await reader.readexactly(length)

            # Echo back the same packet
            writer.write(length_bytes + payload)
            await writer.drain()
    except asyncio.IncompleteReadError:
        print("[INFO] Client disconnected.")
    finally:
        writer.close()
        await writer.wait_closed()

def main(host: str = "127.0.0.1", port: int = 5001):
    loop = asyncio.get_event_loop()
    server = loop.run_until_complete(asyncio.start_server(handle_client, host, port))
    print(f"[INFO] Echo server running on {host}:{port}")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()


if __name__ == "__main__":
    main()