import re, pathlib

root = pathlib.Path("/home/thelegendiv/finn")

NEW_FILES = (
    '            rtllib_dir + "mvu_pkg.sv",\n'
    '            rtllib_dir + "mvu_vvu_axi.sv",\n'
    '            rtllib_dir + "replay_buffer.sv",\n'
    '            rtllib_dir + "mvu.sv",\n'
    '            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",\n'
    '            rtllib_dir + "add_multi.sv",\n'
)
OLD_FILES = (
    '            rtllib_dir + "mvu_vvu_axi.sv",\n'
    '            rtllib_dir + "replay_buffer.sv",\n'
    '            rtllib_dir + "mvu_4sx4u.sv",\n'
    '            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",\n'
    '            rtllib_dir + "mvu_8sx8u_dsp48.sv",\n'
)

def sub1(text, old, new, what):
    assert text.count(old) == 1, f"{what}: expected 1 match, got {text.count(old)}"
    return text.replace(old, new)

# MVAU_rtl
p = root / "src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py"
t = p.read_text()
t = sub1(t, OLD_FILES, NEW_FILES, "mvau files")
old_resolve = t[t.index("    def _resolve_impl_style(self, dsp_block):"):t.index("    def generate_hdl(self, model, fpgapart, clk):")]
new_resolve = '''    def _resolve_dsp_version(self, dsp_block):
        # Based on target device, choose the DSP generation for the RTL compute core
        # (1: DSP48E1, 2: DSP48E2, 3: DSP58)
        assert (
            self.get_nodeattr("resType") != "lut"
        ), """LUT-based RTL-MVU implementation currently not supported!
        Please change resType for {} to 'dsp' or consider switching to HLS-based MVAU!""".format(
            self.onnx_node.name
        )

        match dsp_block:
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

'''
t = t.replace(old_resolve, new_resolve)
t = sub1(t, 'code_gen_dict["$COMPUTE_CORE$"] = [self._resolve_impl_style(dsp_block)]',
         'code_gen_dict["$VERSION$"] = [str(self._resolve_dsp_version(dsp_block))]', "mvau codegen")
p.write_text(t)

# VVAU_rtl
p = root / "src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py"
t = p.read_text()
t = sub1(t, OLD_FILES, NEW_FILES, "vvau files")
t = sub1(t, "def _resolve_impl_style(self, fpgapart):", "def _resolve_dsp_version(self, fpgapart):", "vvau def")
t = sub1(t, '        return "mvu_vvu_8sx9_dsp58"\n', "        return 3\n", "vvau return")
t = sub1(t, 'code_gen_dict["$COMPUTE_CORE$"] = [self._resolve_impl_style(fpgapart)]',
         'code_gen_dict["$VERSION$"] = [str(self._resolve_dsp_version(fpgapart))]', "vvau codegen")
p.write_text(t)

# wrapper template (keep local port names and FORCE_BEHAVIORAL param)
p = root / "finn-rtllib/mvu/mvu_vvu_axi_wrapper.v"
t = p.read_text()
t = sub1(t, 'parameter\tCOMPUTE_CORE = "$COMPUTE_CORE$",', "parameter\tVERSION = $VERSION$,\t// 1: DSP48E1, 2: DSP48E2, 3: DSP58", "wrapper param")
t = sub1(t, ".COMPUTE_CORE(COMPUTE_CORE)", ".VERSION(VERSION)", "wrapper inst")
p.write_text(t)
print("python/wrapper edits ok")
