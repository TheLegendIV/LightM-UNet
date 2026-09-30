import ast
import py_compile
import sys

path = "/home/thelegendiv/finn/notebooks/enet/finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3.py"
ast.parse(open(path).read())
py_compile.compile(path, doraise=True)
print("SYNTAX_OK")
