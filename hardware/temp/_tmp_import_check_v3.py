import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
sys.argv = ["x"]
import importlib
m = importlib.import_module("finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3")
print("IMPORT_OK")
print("FOLDING_BLOCK_FILE=", m.FOLDING_BLOCK_FILE)
import os
print("exists:", os.path.exists(m.FOLDING_BLOCK_FILE))
print("THRESH_DISTRIBUTED_BRAM_TRIGGER=", m.THRESH_DISTRIBUTED_BRAM_TRIGGER)
print("FIFO_URAM_DEPTH_THRESHOLD=", m.FIFO_URAM_DEPTH_THRESHOLD)
