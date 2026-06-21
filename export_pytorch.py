#!/usr/bin/env python3
"""
export_pytorch.py - Create a small, portable PyTorch artifact from a training checkpoint.

Training checkpoints (best_model.pth) are large (~160 MB): they carry optimizer
state, scheduler state, loss/accuracy histories, the raw model weights AND an
argparse.Namespace. None of that is needed for deployment.

This script strips a checkpoint down to a self-contained artifact that holds only:
  * config           - model_name, num_classes, input_height/width, num_frames
  * model_state_dict - the deployed (EMA, if present) weights
  * normalization    - ImageNet mean/std used at train time
  * class_names      - ['no_action', 'action']

The result is a plain dict saved with torch.save -> loadable on a Mac with just
`torch`+`timm` (no training code, no argparse). Push/pull it via git or scp, then
run export_executorch.py on it to produce the CPU / CoreML / mlpackage formats.

Usage:
  python export_pytorch.py trainings2/exp07_fastvit_sa12/.../best_model.pth
  python export_pytorch.py <ckpt> --output models/fastvit_sa12.pt
  python export_pytorch.py <ckpt> --use-raw      # use raw (non-EMA) weights
  python export_pytorch.py <ckpt> --torchscript  # also emit a CPU TorchScript .pt
"""

import argparse
import pathlib
import sys

import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["no_action", "action"]


def build_portable(checkpoint_path: pathlib.Path, use_raw: bool) -> dict:
    """Load a training checkpoint and return a slim, portable artifact dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if "args" not in ckpt:
        raise ValueError("Checkpoint has no 'args'; cannot determine model configuration.")
    args = ckpt["args"]

    config = {
        "model_name": args.model_name,
        "num_classes": args.num_classes,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "num_frames": getattr(args, "num_frames", 1),
    }

    # Prefer the EMA weights (model_state_dict) which is what we select/deploy.
    # raw_model_state_dict holds the non-EMA weights when --use-raw is requested.
    if use_raw and "raw_model_state_dict" in ckpt:
        state_dict = ckpt["raw_model_state_dict"]
        weights_kind = "raw"
    else:
        state_dict = ckpt["model_state_dict"]
        weights_kind = "ema" if ckpt.get("ema") else "raw"

    # Strip any stray 'module.' / wrapper prefixes so the dict loads cleanly into
    # a bare timm.create_model(...) on the Mac side.
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    artifact = {
        "config": config,
        "model_state_dict": state_dict,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "class_names": CLASS_NAMES,
        "weights_kind": weights_kind,
        "format": "cricshorts-portable-v1",
        # Carry over the metrics this checkpoint was selected on, for traceability.
        "val_f1": ckpt.get("best_val_f1", ckpt.get("val_f1")),
        "val_acc": ckpt.get("best_val_acc", ckpt.get("val_acc")),
        "val_recall": ckpt.get("val_recall"),
        "val_precision": ckpt.get("val_precision"),
        "epoch": ckpt.get("epoch"),
    }

    print("Portable artifact configuration:")
    print(f"  Architecture : {config['model_name']}")
    print(f"  Input size   : {config['input_width']}x{config['input_height']}")
    print(f"  Num classes  : {config['num_classes']}")
    print(f"  Weights      : {weights_kind}")
    print(f"  Val F1/Acc   : {artifact['val_f1']} / {artifact['val_acc']}")
    return artifact


def maybe_export_torchscript(artifact: dict, output_path: pathlib.Path) -> None:
    """Emit a CPU TorchScript .pt (normalization + softmax baked in) for pure-PyTorch use.

    This is a fully self-contained CPU model: load with torch.jit.load and feed an
    NCHW float32 tensor with pixel values in [0, 255]. No timm needed at inference.
    """
    import timm

    config = artifact["config"]
    base = timm.create_model(config["model_name"], pretrained=False, num_classes=config["num_classes"])
    base.load_state_dict(artifact["model_state_dict"])
    base.eval()

    # Fuse reparameterizable branches (FastViT / MobileOne) for a smaller CPU model.
    try:
        from timm.utils import reparameterize_model
        base = reparameterize_model(base).eval()
    except Exception as exc:  # noqa: BLE001 - reparam is best-effort
        print(f"  (reparameterization skipped: {type(exc).__name__}: {exc})")

    mean = torch.tensor(artifact["normalization"]["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(artifact["normalization"]["std"]).view(1, 3, 1, 1)

    class CpuModel(torch.nn.Module):
        def __init__(self, m, mean, std):
            super().__init__()
            self.m = m
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x / 255.0 - self.mean) / self.std
            return torch.nn.functional.softmax(self.m(x), dim=1)

    wrapped = CpuModel(base, mean, std).eval()
    example = torch.rand(1, 3, config["input_height"], config["input_width"]) * 255.0
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, example)

    ts_path = output_path.with_name(output_path.stem + "_cpu_ts.pt")
    scripted.save(str(ts_path))
    size_mb = ts_path.stat().st_size / (1024 * 1024)
    print(f"TorchScript CPU model saved: {ts_path} ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable PyTorch artifact from a training checkpoint.")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to training best_model.pth")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output .pt path. Default: models/<model_name>.pt")
    parser.add_argument("--use-raw", action="store_true",
                        help="Use raw (non-EMA) weights instead of the deployed EMA weights.")
    parser.add_argument("--torchscript", action="store_true",
                        help="Also emit a self-contained CPU TorchScript .pt (needs timm).")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    artifact = build_portable(args.checkpoint, use_raw=args.use_raw)

    if args.output is None:
        output_path = pathlib.Path("models") / f"{artifact['config']['model_name']}.pt"
    else:
        output_path = args.output
        if output_path.suffix != ".pt":
            output_path = output_path.with_suffix(".pt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nPortable PyTorch artifact saved: {output_path} ({size_mb:.2f} MB)")

    if args.torchscript:
        maybe_export_torchscript(artifact, output_path)

    print("\nNext (on the Mac, after pulling the .pt):")
    print(f"  python export_executorch.py {output_path} --compare           # all 3 formats + parity check")
    print(f"  python export_executorch.py {output_path} --backend xnnpack    # CPU .pte")
    print(f"  python export_executorch.py {output_path} --backend coreml     # ExecuTorch CoreML/ANE .pte")
    print(f"  python export_executorch.py {output_path} --backend mlpackage  # native Core ML .mlpackage")


if __name__ == "__main__":
    main()
