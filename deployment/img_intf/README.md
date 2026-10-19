# img_proc_pipeline
Includes image processing pipeline with Python powered PC interface, Vitis and Vivado projects for implementation on UltraScale+ platform. 

- `interfaces/send_image.py`: send one image to the board (also holds `BatchImageStreamer`, the v3 protocol client).
- `interfaces/stream_images.py`: async full-duplex streaming client for the pipelined firmware (`FINN_interface/lwip_stream`, "IMGS" protocol: frames sent back-to-back, results returned in order, ACK only after STOP); `--loop` cycles the inputs until Ctrl-C and then stops cleanly. `utils/stream_sim_server.py` is its localhost model (ring of N slots, TCP backpressure); `test/test_stream_loopback.py` runs both.
- `../collect_results.py`: test-split evaluation on the board (`FINN_interface/lwip_collect_results`, "IMGC" protocol, client `interfaces/collect_split.py`): upload the whole Dataset509 test set into board DDR, per-image latency pass, repeated timed passes over the split (PL ticks from the `axi_late_mon` block, no network), download the class maps, score Dice, upsert `../results.csv`. `utils/collect_sim_server.py` is its localhost model; `test/test_collect_protocol.py` runs both. Live-stream metrics belong to `stream_images.py` + `lwip_stream`.
- `utils/`: preprocessing (`preprocess.py`), PNG rendering (`raw_to_png.py`), header helpers, simulated servers.
- `input/`, `output/<model-name>/raw`: generated per run, git-ignored.
