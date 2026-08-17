import py_compile
py_compile.compile("finn_enet_ip_build_partitioned.py", doraise=True)
py_compile.compile("finn_stage_partition.py", doraise=True)
py_compile.compile("finn_partition_build_steps.py", doraise=True)
py_compile.compile("finn_enet_ip_resume_partitioned.py", doraise=True)
print("all compile OK")
