from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, multilabel_metrics


class CorruptedWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds: PTBXLRecordDataset, noise_std: float = 0.0, mask_ratio: float = 0.0):
        self.base_ds = base_ds
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask_ratio > 0:
            t = x.shape[-1]
            mask_len = max(1, int(t * self.mask_ratio))
            start = np.random.randint(0, max(1, t - mask_len))
            x[:, start : start + mask_len] = 0.0
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTB-XL model under clean/noisy/masked conditions.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)

    base_ds = PTBXLRecordDataset(args.data_root, db_df, labels, splits.test_idx, signal_length=args.signal_length)
    test_ds = CorruptedWrapper(base_ds, noise_std=args.noise_std, mask_ratio=args.mask_ratio)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ECGClassifier(encoder=ECGEncoder1DCNN(in_ch=12, width=64), n_classes=labels.shape[1]).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.append(prob)
            y_true.append(y.numpy())

    y_true_arr = np.concatenate(y_true, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    metrics = multilabel_metrics(y_true_arr, y_prob_arr)
    print(metrics)


if __name__ == "__main__":
    main()
