import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. Matching is done by name, and a key is only
    transferred if the shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the
    segmentation maps, identified by keys containing '.seg_layers.') are never transferred, since they're
    dataset/class-count specific by design.

    Beyond that, any key present in the pretrained checkpoint but shape-mismatched against the current network
    (e.g. a custom architecture's own output layer -- not named 'seg_layers', so not covered by the skip list
    above -- when the number of segmentation classes differs between the pretrained checkpoint's dataset and the
    current one) is skipped with a warning instead of raising, so cross-dataset transfer with a different class
    count doesn't hard-crash on an otherwise-compatible architecture. Keys present in the current network but
    entirely absent from the pretrained checkpoint are likewise skipped (left at their freshly-initialized
    value) with a warning, rather than asserted on.

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Sort every key in the current network into: transferable (name not skipped, present in the
    # pretrained checkpoint, matching shape) vs. skipped (for one of the three reasons below) -- skipped keys
    # keep their freshly-initialized value rather than crashing the load.
    transferable = {}
    skipped_by_name, skipped_missing, skipped_shape = [], [], []
    for key, value in model_dict.items():
        if any(i in key for i in skip_strings_in_pretrained):
            skipped_by_name.append(key)
        elif key not in pretrained_dict:
            skipped_missing.append(key)
        elif pretrained_dict[key].shape != value.shape:
            skipped_shape.append((key, pretrained_dict[key].shape, value.shape))
        else:
            transferable[key] = pretrained_dict[key]

    model_dict.update(transferable)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if skipped_shape:
        print(f"Skipped {len(skipped_shape)} key(s) with a shape mismatch (kept freshly-initialized):")
        for key, pretrained_shape, current_shape in skipped_shape:
            print(f"  {key}: pretrained {pretrained_shape} vs current {current_shape}")
    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} key(s) absent from the pretrained checkpoint (kept freshly-initialized):")
        for key in skipped_missing:
            print(f"  {key}")
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in transferable.items():
            print(key, 'shape', value.shape)
    print("################### Done ###################")
    mod.load_state_dict(model_dict)


