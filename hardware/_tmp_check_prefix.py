from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper("finn_deployment_outputs/tied_dsr104_preamble_20261002_011836/intermediate_models/assign_stage_partition_ids_8way.onnx")
# this is pre-build; check the actual partition-0 kernel checkpoint instead
import glob
for fn in glob.glob("finn_build_tmp/tied_dsr104/GenericPartition_0/*.onnx"):
    print(fn)
