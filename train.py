"""Train a flower classifier on an ImageFolder dataset with train/valid/test splits.

Example:
    python train.py flowers --arch vgg16 --epochs 5 --gpu
"""
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from model import ARCHS, EVAL_TRANSFORM, TRAIN_TRANSFORM, build_model, get_device, head_parameters, save_checkpoint


def load_data(data_dir, batch_size=64, num_workers=2):
    data_dir = Path(data_dir)
    sets = {
        "train": datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM),
        "valid": datasets.ImageFolder(data_dir / "valid", transform=EVAL_TRANSFORM),
        "test": datasets.ImageFolder(data_dir / "test", transform=EVAL_TRANSFORM),
    }
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        for name, ds in sets.items()
    }
    return loaders, sets["train"].class_to_idx


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def train(data_dir, save_path="checkpoint.pth", arch="vgg16", hidden_units=512, lr=1e-3,
          epochs=5, batch_size=64, use_gpu=True, num_workers=2, pretrained=True):
    device = get_device(use_gpu)
    loaders, class_to_idx = load_data(data_dir, batch_size, num_workers)
    model = build_model(arch, hidden_units, len(class_to_idx), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head_parameters(model, arch), lr=lr)

    print(f"Training {arch} on {device} for {epochs} epoch(s), {len(class_to_idx)} classes")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
        valid_loss, valid_acc = evaluate(model, loaders["valid"], criterion, device)
        print(f"Epoch {epoch}/{epochs}  train loss {running_loss / seen:.3f}  "
              f"valid loss {valid_loss:.3f}  valid acc {valid_acc:.1%}")

    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss {test_loss:.3f}  test acc {test_acc:.1%}")
    save_checkpoint(save_path, model, arch, hidden_units, class_to_idx, epochs)
    print(f"Saved checkpoint to {save_path}")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Train a flower image classifier.")
    parser.add_argument("data_dir", help="folder containing train/, valid/ and test/ subfolders")
    parser.add_argument("--save_path", default="checkpoint.pth")
    parser.add_argument("--arch", default="vgg16", choices=sorted(ARCHS))
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", action="store_true", help="use CUDA or Apple MPS if available")
    args = parser.parse_args()

    train(args.data_dir, args.save_path, args.arch, args.hidden_units, args.learning_rate,
          args.epochs, args.batch_size, args.gpu, args.num_workers)


if __name__ == "__main__":
    main()
