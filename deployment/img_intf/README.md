# img_proc_pipeline
Includes image processing pipeline with Python powered PC interface, Vitis and Vivado projects for implementation on UltraScale+ platform. 

- `interfaces/send_image.py`: send one image to the board (also holds `BatchImageStreamer`, the v3 protocol client).
- `interfaces/stream_images.py`: async full-duplex streaming client for the pipelined firmware (`FINN_interface/lwip_stream`, "IMGS" protocol: frames sent back-to-back, results returned in order, ACK only after STOP). `utils/stream_sim_server.py` is its localhost model (ring of N slots, TCP backpressure); `test/test_stream_loopback.py` runs both.
- `../collect_results.py`: stream the whole Dataset509 test set, score Dice, upsert `../results.csv`.
- `utils/`: preprocessing (`preprocess.py`), PNG rendering (`raw_to_png.py`), header helpers, simulated servers.
- `input/`, `output/<model-name>/raw`: generated per run, git-ignored.
