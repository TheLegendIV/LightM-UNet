<<<<<<< HEAD
from qonnx.core.modelwrapper import ModelWrapper
fn = "finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224/intermediate_models/step_build_all_partitions_capped.onnx"
m = ModelWrapper(fn)
print("graph inputs:", [i.name for i in m.graph.input])
print("graph outputs:", [o.name for o in m.graph.output])
=======
from qonnx.core.modelwrapper import ModelWrapper
fn = "finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224/intermediate_models/step_build_all_partitions_capped.onnx"
m = ModelWrapper(fn)
print("graph inputs:", [i.name for i in m.graph.input])
print("graph outputs:", [o.name for o in m.graph.output])
>>>>>>> 1c37749cf21da213659e029bae27ca2f6f8981fe
