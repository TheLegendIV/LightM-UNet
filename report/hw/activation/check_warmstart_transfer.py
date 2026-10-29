"""Pre-flight for `-pretrained_weights` warm starts of the all-PReLU hardware-baseline net: builds the same ENet the training job builds, loads the source checkpoint through nnU-Net's own
load_pretrained_weights, and FAILS (exit 1) unless EVERY parameter and buffer transferred bit-exactly -- conv / BN (incl. running stats) / every PReLU alpha -- with nothing missing from the
source and nothing left over in it. nnU-Net's loader only warns on a mismatch (it silently keeps the fresh init for any key it can't match, e.g. alphas if the decoder_prelu flag differs), so
a wrong warm start would otherwise go unnoticed.

    python report/hw/activation/check_warmstart_transfer.py --source <.../nnUNetTrainerENet_12_dense_prelu_all__nnUNetPlans__2d/fold_0/checkpoint_best.pth> [--no-decoder-prelu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "enet"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--no-decoder-prelu", action="store_true", help="expect an encoder-only PReLU source (default: PReLU in the decoder too)")
    args = p.parse_args()

    net = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False, context_pattern="dense_dilation",
        use_prelu=True, decoder_prelu=not args.no_decoder_prelu,
    )
    src = torch.load(args.source, map_location="cpu", weights_only=False)["network_weights"]
    load_pretrained_weights(net, str(args.source))

    dst = net.state_dict()
    missing = [k for k in dst if k not in src]
    unexpected = [k for k in src if k not in dst]
    differ = [k for k in dst if k in src and not torch.equal(dst[k].cpu(), src[k].cpu())]
    n_prelu = sum(isinstance(m, nn.PReLU) for m in net.modules())
    alpha_keys = [n + ".weight" for n, m in net.named_modules() if isinstance(m, nn.PReLU)]
    alphas_ok = all(k in src and torch.equal(dst[k].cpu(), src[k].cpu()) for k in alpha_keys)
    print(f"transfer check: {len(dst)} model keys | PReLU modules {n_prelu} (alphas bit-exact: {alphas_ok}) | missing from source {len(missing)} | unexpected in source {len(unexpected)} | differing {len(differ)}")
    for label, ks in (("missing", missing), ("unexpected", unexpected), ("differing", differ)):
        for k in ks[:10]:
            print(f"  {label}: {k}")
    ok = not (missing or unexpected or differ) and alphas_ok and n_prelu > 0
    print("OK: full warm start (conv, BN, every PReLU alpha)" if ok else "FAIL: warm start would NOT carry every parameter")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
