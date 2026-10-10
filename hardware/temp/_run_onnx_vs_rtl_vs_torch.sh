cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
python3 _onnx_vs_rtl_vs_torch.py 0,1,2,3,5,26 2>&1 | grep -E '^case|Error|Traceback' | cut -c1-400
