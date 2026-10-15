"""End-to-end smoke test: train on a tiny synthetic dataset, reload, predict.

Runs on CPU in a few seconds with random weights (no ImageNet download).
"""
import json

from PIL import Image

from model import get_device, load_checkpoint, predict
from train import train

# Folder names sort as strings ("1" < "10" < "2"), so index != int(name) - 1.
CLASSES = ["1", "10", "2"]
COLORS = {"1": (255, 0, 0), "10": (0, 255, 0), "2": (0, 0, 255)}


def make_dataset(root):
    for split in ("train", "valid", "test"):
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (64, 64), COLORS[cls]).save(folder / f"{i}.jpg")


def test_train_save_load_predict(tmp_path):
    make_dataset(tmp_path / "data")
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"1": "rose", "10": "tulip", "2": "daisy"}))
    ckpt = tmp_path / "ckpt.pth"

    acc = train(tmp_path / "data", ckpt, arch="densenet121", hidden_units=16, epochs=1,
                batch_size=4, use_gpu=False, num_workers=0, pretrained=False)
    assert 0.0 <= acc <= 1.0

    device = get_device(False)
    model, class_to_idx = load_checkpoint(ckpt, device)
    assert class_to_idx == {"1": 0, "10": 1, "2": 2}

    results = predict(tmp_path / "data" / "test" / "10" / "0.jpg", model, class_to_idx, device,
                      topk=3, category_names=names)
    assert len(results) == 3
    assert {label for label, _ in results} == {"rose", "tulip", "daisy"}
    assert abs(sum(p for _, p in results) - 1.0) < 1e-4
