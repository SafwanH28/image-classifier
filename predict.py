"""Predict the most likely flower species for an image.

Example:
    python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3 --gpu
"""
import argparse

from model import get_device, load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description="Classify a flower image with a trained checkpoint.")
    parser.add_argument("image", help="path to the image")
    parser.add_argument("checkpoint", nargs="?", default="checkpoint.pth")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="use CUDA or Apple MPS if available")
    args = parser.parse_args()

    device = get_device(args.gpu)
    model, class_to_idx = load_checkpoint(args.checkpoint, device)
    for label, prob in predict(args.image, model, class_to_idx, device, args.top_k, args.category_names):
        print(f"{prob:6.1%}  {label}")


if __name__ == "__main__":
    main()
