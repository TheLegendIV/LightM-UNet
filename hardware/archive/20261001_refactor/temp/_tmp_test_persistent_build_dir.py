import os
os.environ["FINN_BUILD_DIR"] = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp"
import sys
sys.path.insert(0, "/home/thelegendiv/finn/src")
from finn.util.basic import make_build_dir
d = make_build_dir(prefix="probe_persistent_builddir_")
print("created:", d)
print("exists:", os.path.exists(d))
