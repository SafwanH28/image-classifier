"""Model building, checkpointing and inference for the flower classifier."""
import json

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# arch -> (constructor, pretrained weights, name of the head attribute, head input size)
ARCHS = {
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT, "classifier", 25088),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier", 1024),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc", 2048),
}

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch="vgg16", hidden_units=512, num_classes=102, dropout=0.2, pretrained=True):
    """Pretrained backbone with frozen features and a new trainable classifier head."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}', choose from {sorted(ARCHS)}")
    constructor, weights, head_name, in_features = ARCHS[arch]
    model = constructor(weights=weights if pretrained else None)
    for param in model.parameters():
        param.requires_grad = False
    setattr(model, head_name, nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, num_classes),
    ))
    return model


def head_parameters(model, arch):
    return getattr(model, ARCHS[arch][2]).parameters()


def save_checkpoint(path, model, arch, hidden_units, class_to_idx, epochs):
    torch.save({
        "arch": arch,
        "hidden_units": hidden_units,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "epochs": epochs,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # Skip the ImageNet download: every weight is overwritten by the checkpoint.
    model = build_model(checkpoint["arch"], checkpoint["hidden_units"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["class_to_idx"]


def predict(image_path, model, class_to_idx, device, topk=5, category_names=None):
    """Return the top-k (label, probability) pairs for one image."""
    image = EVAL_TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)
    top_p, top_idx = probs.topk(min(topk, probs.shape[1]), dim=1)

    # ImageFolder sorts class folders as strings ("1", "10", "100", ...), so the
    # output index has to be mapped back through class_to_idx, not used directly.
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    names = {}
    if category_names:
        with open(category_names) as f:
            names = json.load(f)
    labels = [names.get(idx_to_class[i], idx_to_class[i]) for i in top_idx[0].tolist()]
    return list(zip(labels, top_p[0].tolist()))
