import re
path = "/home/thelegendiv/finn/notebooks/enet/vivado_projects/zcu7ev_S12_dense_256_u4_analytical_v1/zcu7ev_S12_dense_256_u4_analytical_v1.xpr"
with open(path, "r", errors="replace") as f:
    data = f.read()
idx = data.lower().find("ip_repo_paths")
if idx == -1:
    print("ip_repo_paths NOT FOUND in xpr")
else:
    print(data[max(0, idx-200):idx+2000])
