#!/bin/bash
F=/tmp/p6_final_state.txt
echo "=== fork/join/upsample/pad/SWG ports ==="
grep -E ' (AddStreams_hls_[012]\.(in0_V|in1_V|out_V)|DuplicateStreams_hls_[012]\.(in0_V|out0_V|out1_V)|FMPadding_Pixel_hls_0\.(in0_V|out_V)|ConvolutionInputGenerator_rtl_[01]\.(in0_V|out_V)|UpsampleNearestNeighbour_hls_0\.(in0_V|out_V))$' $F | sort -k5
echo
echo "=== FIFO output-side state tally ==="
grep -E 'StreamingFIFO_rtl_[0-9]+\.out_V$' $F | awk '{print $3}' | sort | uniq -c
echo
echo "=== FIFOs with out V=1 R=0 (holding data, blocked) ==="
grep -E 'StreamingFIFO_rtl_[0-9]+\.out_V$' $F | grep 'V=1 R=0' | awk '{print $NF}' | tr '\n' ' '
