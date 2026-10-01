p='/workspace/LightM-UNet/compression/slurm/qat_12_dense_relu_nearest_upsample_wm_lex_dsr2_fps305_candidatebits468_ft15ep.job'
s=open(p).read()
a=s.index('    for EP in 5 10 15; do'); b=s.index('    done\n',a)+len('    done\n')
s=s[:a]+'    # Inference ONLY on checkpoint_best.pth for QAT jobs (epoch5/10/15 snapshots are deliberately not evaluated).\n'+s[b:]
s=s.replace('''    if [ "$BEST_EPOCH" != "$FINAL_EPOCH" ]; then
        echo "NOTE: 'best' EMA dice plateaued before the run finished (last improved at epoch ${BEST_EPOCH}, training reached ${FINAL_EPOCH}) -- the standard row below reports epoch-${BEST_EPOCH} weights, not epoch-${FINAL_EPOCH}. This is real, not a failure -- the epoch5/10/15 rows below are NOT affected (fixed snapshots, independent of EMA-best tracking)."
    fi''','''    if [ "$BEST_EPOCH" != "$FINAL_EPOCH" ]; then
        echo "NOTE: 'best' EMA dice plateaued before the run finished (last improved at epoch ${BEST_EPOCH}, training reached ${FINAL_EPOCH}) -- the row below reports epoch-${BEST_EPOCH} weights, not epoch-${FINAL_EPOCH}. This is real, not a failure."
    fi''')
open(p,'w',newline='\n').write(s)
