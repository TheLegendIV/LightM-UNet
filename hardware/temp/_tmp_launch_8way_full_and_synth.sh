cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_finn_calibrated_8way_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_finn_calibrated_preamble_20260914_180444 /home/thelegendiv/finn/notebooks/enet/$OUTDIR && python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR" > /home/thelegendiv/finn/notebooks/enet/$OUTDIR/full_build_and_synth.log 2>&1 &
disown
echo "LAUNCHED_PID=$!"
echo "OUTDIR=$OUTDIR"
