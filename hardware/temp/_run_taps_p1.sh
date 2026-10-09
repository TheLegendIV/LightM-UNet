cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
nohup python3 rtlsim_taps.py --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200 --partition 1 --input /tmp/golden_pp/case0/p1_in.raw --out /tmp/taps_p1 > /tmp/taps_p1.log 2>&1 &
echo started $!
