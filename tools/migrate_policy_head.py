import argparse
from pathlib import Path
import torch

POLICY_SIZE = 4096  # Target output size


def migrate_checkpoint(src_path: Path, dst_path: Path):
    """Load *src_path* checkpoint, reshape policy head to 4096 outputs, save to *dst_path*.

    The function handles both state-dict-only checkpoints and composite dicts
    containing ``model_state_dict`` + metadata.  We simply truncate or pad the
    final linear layer in ``policy_head`` to match *POLICY_SIZE*.
    """
    ckpt = torch.load(src_path, map_location="cpu")

    # Resolve state dict reference
    state_dict_key = "model_state_dict" if isinstance(ckpt, dict) and "model_state_dict" in ckpt else None
    state_dict = ckpt[state_dict_key] if state_dict_key else ckpt

    # Locate final linear layer of policy head
    linear_w_name = None
    # Heuristic: policy head is the *last 2-D weight* whose second dim equals 4096 (input features)
    candidates = [(n, t) for n, t in state_dict.items() if t.ndim == 2]
    for name, tensor in candidates[::-1]:  # iterate reversed -> later layers first
        if tensor.shape[1] == POLICY_SIZE or ".policy_head" in name:
            linear_w_name = name
            break
    if linear_w_name is None:
        raise RuntimeError("Could not locate policy head Linear weight in checkpoint")

    linear_b_name = linear_w_name.replace(".weight", ".bias")

    W = state_dict[linear_w_name]
    B = state_dict[linear_b_name]

    in_features = W.shape[1]
    out_features = W.shape[0]

    if out_features == POLICY_SIZE:
        print(f"{src_path.name}: already at {POLICY_SIZE}, skipping")
        return

    if out_features > POLICY_SIZE:
        # Truncate
        W_new = W[:POLICY_SIZE, :].clone()
        B_new = B[:POLICY_SIZE].clone()
        print(f"{src_path.name}: truncated policy head from {out_features} to {POLICY_SIZE}")
    else:
        # Pad with zeros
        pad = POLICY_SIZE - out_features
        W_new = torch.zeros(POLICY_SIZE, in_features)
        B_new = torch.zeros(POLICY_SIZE)
        W_new[:out_features] = W
        B_new[:out_features] = B
        print(f"{src_path.name}: padded policy head from {out_features} to {POLICY_SIZE}")

    state_dict[linear_w_name] = W_new
    state_dict[linear_b_name] = B_new

    # Update metadata config if present
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        ckpt["model_config"]["policy_head_output_size"] = POLICY_SIZE

    # Save
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt if state_dict_key else state_dict, dst_path)
    print(f"✅ Saved migrated checkpoint to {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoints to 4096-output policy head")
    parser.add_argument("input", type=str, help="Path to original .pt checkpoint")
    parser.add_argument("output", type=str, nargs="?", help="Destination path (default: append _4096.pt)")
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve() if args.output else src.with_stem(src.stem + "_4096")

    migrate_checkpoint(src, dst)


if __name__ == "__main__":
    main() 