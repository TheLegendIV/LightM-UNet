from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
print([m for m in dir(HWCustomOp) if 'cycle' in m.lower() or 'exp' in m.lower()])
