from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import SSLReconstructionModel
from ssrl_ecg.utils import apply_random_mask, choose_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised ECG pretraining with masked reconstruction.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_encoder.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)

    train_ds = PTBXLRecordDataset(
        data_root=args.data_root,
        db_df=db_df,
        labels=labels,
        indices=splits.train_idx,
        use_high_resolution=False,
        signal_length=args.signal_length,
        return_labels=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SSLReconstructionModel(in_ch=12, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"SSL Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            x_masked = apply_random_mask(x, mask_ratio=args.mask_ratio)

            optimizer.zero_grad(set_to_none=True)
            x_hat = model(x_masked)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(1, n):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict()}, args.out)
    print(f"Saved SSL encoder checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
