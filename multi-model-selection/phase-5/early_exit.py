# branchynet_mnist.py
# Minimal BranchyNet (early-exit) demo on MNIST with PyTorch
# - Two early exits + final head
# - Joint training (weighted sum of CE losses)
# - Early-exit during evaluation by confidence threshold
# - Reports accuracy per exit and exit distribution
#
# Usage (CPU or CUDA):
#   python branchynet_mnist.py --epochs 2 --batch-size 128 --thresh 0.9
#   python branchynet_mnist.py --epochs 2 --batch-size 128 --thresh 0.8

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# Model: simple CNN + branches
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)

class ExitHead(nn.Module):
    """A lightweight classifier head attached to an intermediate feature map."""
    def __init__(self, in_ch, num_classes=10):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, feat):
        return self.head(feat)

class BranchyNet(nn.Module):
    """
    Backbone:
      Block1: 1->32
      Block2: 32->64
      Block3: 64->128
    Early Exits:
      Exit1 after Block1
      Exit2 after Block2
    Final Head after Block3
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)

        self.exit1 = ExitHead(32, num_classes)
        self.exit2 = ExitHead(64, num_classes)
        self.final_head = ExitHead(128, num_classes)

    def forward(self, x):
        """
        Returns list of logits from each head in order [exit1, exit2, final]
        plus intermediate features (optional).
        """
        f1 = self.block1(x)           # [B,32,14,14]
        f2 = self.block2(f1)          # [B,64,7,7]
        f3 = self.block3(f2)          # [B,128,3,3]

        l1 = self.exit1(f1)
        l2 = self.exit2(f2)
        l3 = self.final_head(f3)
        return [l1, l2, l3]


# ----------------------------
# Training / Evaluation helpers
# ----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 2
    batch_size: int = 128
    exit_loss_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)

def loss_all_exits(logits_list: List[torch.Tensor], targets: torch.Tensor,
                   weights: Tuple[float, float, float]) -> torch.Tensor:
    losses = []
    for w, logits in zip(weights, logits_list):
        losses.append(w * F.cross_entropy(logits, targets))
    return sum(losses)

@torch.no_grad()
def evaluate_with_early_exit(model: nn.Module, loader: DataLoader, device: torch.device,
                             threshold: float = 0.9) -> Dict:
    """
    Early-exit policy: For each sample, take the earliest head whose confidence
    (max softmax probability) >= threshold. Otherwise fall through to final.
    """
    model.eval()
    total = 0
    correct = 0
    correct_per_exit = [0, 0, 0]
    count_per_exit = [0, 0, 0]

    # Simple compute cost proxy: fraction of blocks used (1/3, 2/3, 3/3)
    cost_per_exit = [1/3, 2/3, 1.0]
    total_cost = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits_list = model(x)  # [exit1, exit2, final]
        batch_size = x.size(0)

        # For each sample, decide which exit to use
        probs_list = [F.softmax(l, dim=1) for l in logits_list]
        conf_list, pred_list = zip(*[(p.max(dim=1).values, p.argmax(dim=1)) for p in probs_list])

        for i in range(batch_size):
            exit_idx = 2  # default final
            for k in range(3):  # check exit1, exit2, final (final always qualifies implicitly)
                if k < 2:
                    if conf_list[k][i].item() >= threshold:
                        exit_idx = k
                        break
                else:
                    exit_idx = 2

            pred = pred_list[exit_idx][i].item()
            tgt = y[i].item()
            total += 1
            count_per_exit[exit_idx] += 1
            if pred == tgt:
                correct += 1
                correct_per_exit[exit_idx] += 1
            total_cost += cost_per_exit[exit_idx]

    acc = correct / total if total else 0.0
    avg_cost = total_cost / total if total else 0.0

    exit_stats = {
        "accuracy_overall": acc,
        "avg_compute_cost_proxy": avg_cost,
        "exit_counts": {
            "exit1": count_per_exit[0],
            "exit2": count_per_exit[1],
            "final": count_per_exit[2],
        },
        "accuracy_per_exit": {
            "exit1": (correct_per_exit[0] / count_per_exit[0]) if count_per_exit[0] else 0.0,
            "exit2": (correct_per_exit[1] / count_per_exit[1]) if count_per_exit[1] else 0.0,
            "final": (correct_per_exit[2] / count_per_exit[2]) if count_per_exit[2] else 0.0,
        }
    }
    return exit_stats


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--thresh", type=float, default=0.9, help="Early-exit confidence threshold")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),                   # [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST stats
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=use_cuda)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)

    # Model
    model = BranchyNet(num_classes=10).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

    # Training
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits_list = model(x)
            loss = loss_all_exits(logits_list, y, weights=(0.3, 0.3, 0.4))
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{cfg.epochs}] Train loss: {epoch_loss:.4f}")

        # Evaluate normally (no early exit) for reference
        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits_list = model(x)
                logits = logits_list[-1]  # final head only
                pred = logits.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
            ref_acc = correct / total
        print(f"  Reference (final-head only) accuracy: {ref_acc:.4f}")

        # Early-exit evaluation
        stats = evaluate_with_early_exit(model, test_loader, device, threshold=args.thresh)
        print(f"  Early-exit accuracy: {stats['accuracy_overall']:.4f}")
        print(f"  Exit distribution: {stats['exit_counts']}")
        print(f"  Accuracy per exit: {stats['accuracy_per_exit']}")
        print(f"  Avg compute cost (proxy 1/3,2/3,1.0): {stats['avg_compute_cost_proxy']:.3f}")

    print("\nDone. Try different --thresh values (e.g., 0.7, 0.8, 0.95) to trade accuracy vs. compute.")

if __name__ == "__main__":
    main()
