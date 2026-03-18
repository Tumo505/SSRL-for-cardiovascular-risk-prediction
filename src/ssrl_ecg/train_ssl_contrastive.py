from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, set_seed


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=1)


class SSLContrastiveModel(nn.Module):
    def __init__(self, in_ch: int = 12, width: int = 64, proj_dim: int = 128):
        super().__init__()
        self.encoder = ECGEncoder1DCNN(in_ch=in_ch, width=width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj_head = ContrastiveProjectionHead(self.encoder.out_channels, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        pooled = self.pool(z).squeeze(-1)
        proj = self.proj_head(pooled)
        return proj


def random_augment_signal(x: torch.Tensor, aug_type: str = "mix") -> torch.Tensor:
    """Apply simple augmentations: noise, time-shift, scaling."""
    if aug_type == "noise":
        return x + torch.randn_like(x) * 0.05
    elif aug_type == "scale":
        scale = torch.rand((x.shape[0], 1, 1), device=x.device) * 0.2 + 0.9
        return x * scale
    elif aug_type == "shift":
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            return torch.cat([torch.zeros(x.shape[0], x.shape[1], shift, device=x.device), x[:, :, :-shift]], dim=2)
        elif shift < 0:
            return torch.cat([x[:, :, -shift:], torch.zeros(x.shape[0], x.shape[1], -shift, device=x.device)], dim=2)
    return x


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)

        similarity = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = torch.cat([torch.cat([~mask, torch.ones(batch_size, batch_size, dtype=torch.bool, device=z.device)], dim=1),
                          torch.cat([torch.ones(batch_size, batch_size, dtype=torch.bool, device=z.device), ~mask], dim=1)], dim=0)

        pos_mask = torch.cat([torch.eye(batch_size, dtype=torch.bool, device=z.device), torch.eye(batch_size, dtype=torch.bool, device=z.device)], dim=0)
        pos_idx = torch.cat([torch.arange(batch_size, device=z.device) + batch_size, torch.arange(batch_size, device=z.device)], dim=0)

        loss = 0.0
        for i in range(2 * batch_size):
            pos_sim = similarity[i, pos_idx[i]]
            neg_sim = similarity[i, mask[i]]
            loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum()))
        return loss / (2 * batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL pretraining with contrastive learning (NT-Xent).")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_contrastive_encoder.pt"))
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

    model = SSLContrastiveModel(in_ch=12, width=64, proj_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = NTXentLoss(temperature=args.temperature)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Contrastive Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            x_aug1 = random_augment_signal(x, aug_type="noise")
            x_aug2 = random_augment_signal(x, aug_type="scale")

            optimizer.zero_grad(set_to_none=True)
            z_i = model(x_aug1)
            z_j = model(x_aug2)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(1, n):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict()}, args.out)
    print(f"Saved contrastive SSL encoder checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
