import json,glob,onnx
tag='dsr5.0'
base='/workspace/LightM-UNet/'
mf=glob.glob(base+f'MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/dsrSweep_pbiOff_dsr{tag[3:]}/layer_bits_folding_*.json')[0]
pl=json.load(open(mf))['per_layer']
names=list(pl); start=names.index('stage2.0.reduce.0')
lay=[(k,pl[k]) for k in names[start:] if pl[k].get('mvu_cycles') is not None]
m=onnx.load(base+f'hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/post_fifo_autosize_checkpoints/partition2_dsrSweep_pbiOff_dsr{tag[3:]}_milpfold_prefifo_autosize.onnx')
def at(n): return {x.name:x.i for x in n.attribute if x.type==2}
mv=[(n.name,at(n)) for n in m.graph.node if n.op_type.startswith('MVAU')]
sw=[(n.name,at(n)) for n in m.graph.node if n.op_type.startswith('ConvolutionInputGenerator')]
print('MILP conv-like layers from stage2.0:',len(lay),'ONNX MVAU',len(mv),'SWG',len(sw))
bad=0
for (k,e),(n,a) in zip(lay,mv):
    ok=(e['pe'],e['simd'],e['mvu_cycles'])==(a.get('PE'),a.get('SIMD'),a.get('cycles_estimate'))
    bad+=not ok
    if not ok: print('MVAU MISMATCH',k,(e['pe'],e['simd'],e['mvu_cycles']),n,(a.get('PE'),a.get('SIMD'),a.get('cycles_estimate')))
print('MVAU mismatches',bad,'of',min(len(lay),len(mv)))
swl=[(k,e) for k,e in lay if e.get('swu_cycles') and e.get('simd_swu') and e['swu_cycles']>0 and k.endswith('conv') or k.endswith('conv.0')]
print('SWG (MILP simd_swu/swu_cycles  vs ONNX SIMD/cycles):')
for (k,e),(n,a) in zip([x for x in lay[:len(mv)] if x[0].endswith(("conv","conv.0"))],sw):
    print(' ',k,e['simd_swu'],e['swu_cycles'],'|',n,a.get('SIMD'),a.get('cycles_estimate'))
print('MILP max cycles in these 15 layers (mvu):',max(e['mvu_cycles'] for k,e in lay[:len(mv)]),'| max MILP swu:',max(e['swu_cycles'] for k,e in lay[:len(mv)]))
