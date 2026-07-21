# LM-UNet Memory Profiling Scripts

This folder contains two analytical memory estimators for the LM-UNet implementation in
`nnunetv2/nets/LMUNet.py`.

The scripts do not measure GPU allocator memory. They estimate architectural tensor sizes for
FPGA-oriented reasoning.

## 1. Static Tensor Memory

Script:

```bash
python memory_profiles/tensor_memory.py
```

Purpose:

```text
List the static architecture tensors in LM-UNet.
```

It answers:

```text
How large is f1?
How large are the skip tensors?
How large is the edge map?
How large are decoder outputs and logits?
How much memory do weights/biases/LayerNorm/Mamba parameters require?
```

For a tensor with shape `[B, C, H, W]`, memory is:

```text
bytes = B * C * H * W * bytes_per_element
```

For a trainable parameter tensor, memory is:

```text
bytes = number_of_parameters * parameter_bytes_per_element
```

Parameter records are counted from the actual `LMUNet.named_parameters()` output, so this includes:

```text
Conv weights and biases
GroupNorm/LayerNorm affine parameters
Mamba projection weights
Mamba A_log and D parameters
PV-Mamba alpha parameters
final classifier weights and biases
```

Default settings match the current ARCADE 2D LM-UNet configuration:

```text
input:        [1, 1, 512, 512]
classes:      4
channels:     12,20,32,44,64,72
edge_channels: 20
dtype:        fp32
```

This script groups tensors into:

```text
main path:
  input, encoder outputs, decoder outputs, final logits

skip path:
  raw encoder features retained for decoder skips
  refined skip features produced by MMSC

edge path:
  EFE low/high tensors, projected high tensor, edge map, resized edge maps
```

Important:

```text
This is the architecture tensor catalog.
It includes feature maps and trainable parameters.
It does not model extra live working buffers needed during computation.
```

## 2. Working/Dynamic Memory Estimate

Script:

```bash
python memory_profiles/working_memory.py
```

Purpose:

```text
Estimate extra live tensors needed while computing the static tensors from report 1.
```

Assumptions:

```text
1. The static feature maps and parameters are already listed in tensor_memory.py.
2. In-place processing is assumed where possible.
3. Residual adds require the original tensor and transformed tensor to be live at the same time.
4. Concats require both input tensors and the concatenated tensor to be live at the same time.
5. Attention gates require the feature tensor plus the gate tensor to be live at the same time.
6. Approximate Mamba internals are listed as full-sequence upper-bound working tensors.
7. Trainable parameters are excluded here because they are already in the static tensor report.
```

This script separates:

```text
local working peaks:
  residual buffers, decoder concat buffers, EFE concat, MMSC attention buffers

peak working memory by stage/module:
  the largest local working row per encoder stage, EFE, EFF/MMSA stage, and decoder stage

approximate Mamba working tensors:
  flattened token tensors and projection-like intermediates
```

This working-memory estimate is the better number for:

```text
Which operation creates the largest extra live tensor set?
Which stage has the largest peak working memory?
Which stages are most likely to need tiling, buffering, recomputation, or off-chip spill?
```

Important:

```text
This report is separate from the static tensor catalog.
It does not recount raw skip tensors as persistent storage.
It only estimates additional live tensors required during compute operations.
```

The local working peak estimate is the better number for:

```text
Which module is hardest to keep fully on chip without tiling?
```

## Common Options

Both scripts accept:

```bash
--height 512
--width 512
--batch 1
--in-channels 1
--out-channels 4
--channels 12,20,32,44,64,72
--edge-channels 20
--dtype fp32
--param-dtype fp32
```

Supported dtypes:

```text
fp32 = 4 bytes per element
fp16 = 2 bytes per element
int8 = 1 byte per element
```

## Example

```bash
cd enet
python memory_profiles/tensor_memory.py --dtype fp32
python memory_profiles/working_memory.py --dtype fp32
```
