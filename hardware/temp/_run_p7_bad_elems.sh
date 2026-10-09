cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
nohup bash -c "python3 _p7_bad_elems.py > /tmp/p7_bad_elems.log 2>&1; echo done >> /tmp/p7_bad_elems.log" > /dev/null 2>&1 &
