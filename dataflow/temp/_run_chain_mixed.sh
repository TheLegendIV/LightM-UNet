cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
python3 -c "import numpy as np;d=np.load('quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz');print({k:(d[k].shape,d[k].dtype) for k in d.files})" > /tmp/chain_mixed.log 2>&1
nohup python3 _chain_mixed.py --old finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200 \
  --new-p1 finn_build_tmp/S12_dense_256_u4_u8in_int6_p1fix \
  --ref quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz \
  --cases 0,1 >> /tmp/chain_mixed.log 2>&1 &
