cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_1
F=$(ls -d vivado_stitch_proj_*)/GenericPartition_1_wrapper.v
echo "module GenericPartition_1 decl:"
grep -n -E "^module GenericPartition_1 *[(#]|^module GenericPartition_1$" $F
grep -n -E "^module GenericPartition_1_wrapper" $F
echo "tdata net count (any):"
grep -c -E "^\s*wire .*_TDATA;" $F
grep -n -E "^\s*wire .*_TDATA;" $F | head -8
grep -n -E "^\s*wire .*_TDATA;" $F | tail -4
echo "distinct suffix patterns:"
grep -E "^\s*wire .*_TDATA;" $F | sed -E 's/.*GenericPartition_1_//; s/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -20
echo "first line of module containing line 34417:"
awk 'NR<=34417 && /^module /{l=NR": "$0} END{print l}' $F
