# img_proc_pipeline
Includes image processing pipeline with Python powered PC interface, Vitis and Vivado projects for implementation on UltraScale+ platform. 

- `interfaces/send_image.py`: send one image to the board (also holds `BatchImageStreamer`, the v3 protocol client).
- `../collect_results.py`: stream the whole Dataset509 test set, score Dice, upsert `../results.csv`.
- `utils/`: preprocessing (`preprocess.py`), PNG rendering (`raw_to_png.py`), header helpers, simulated servers.
- `input/`, `output/<model-name>/raw`: generated per run, git-ignored.
