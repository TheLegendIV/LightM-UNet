cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
python3 _dump_partition_nodes.py 1 > /tmp/p1_nodes.txt 2>&1
awk '{print $1}' /tmp/p1_nodes.txt | sort | uniq -c
echo ----
grep -v -E '^(StreamingDataWidthConverter_rtl|Thresholding_rtl|DuplicateStreams_hls|AddStreams_hls|MVAU_rtl|FMPadding_rtl|ConvolutionInputGenerator_rtl) ' /tmp/p1_nodes.txt | cut -c1-260
echo ----
grep -E "^ConvolutionInputGenerator_rtl" /tmp/p1_nodes.txt | cut -c1-300 | grep -E "Stride': \[2|SIMD': 1,"
