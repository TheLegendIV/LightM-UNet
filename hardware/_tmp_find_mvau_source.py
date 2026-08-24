import inspect
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
print(inspect.getsourcefile(MVAU_hls))
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
print(inspect.getsourcefile(MVAU))
