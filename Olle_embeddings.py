import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from clip import load, tokenize  # assumes your local clip/ is on PYTHONPATH
from PIL import Image

def compute_and_save_embeddings(
    data_root: str = "./data",
    batch_size: int = 64,
    model_name: str = "ViT-B/32",
    device: str = None,
    output_dir: str = "./embeddings"
):
    # pick cpu or cuda
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # load CLIP model + preprocessor
    model, preprocess = load(model_name, device=device)
    model.eval()

    # ─── IMAGE EMBEDDINGS ─────────────────────────────────────────────────────────
    mnist_ds = MNIST(root=data_root, train=True, download=True,
                     transform=lambda img: preprocess(img.convert("RGB")))
    loader = DataLoader(mnist_ds, batch_size=batch_size, shuffle=False)

    all_img_embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            embs = model.encode_image(imgs).cpu()
            all_img_embs.append(embs)
    all_img_embs = torch.cat(all_img_embs, dim=0).numpy()
    np.save(os.path.join(output_dir, "mnist_image_embeddings.npy"), all_img_embs)
    print(f"Saved image embeddings → {all_img_embs.shape} to mnist_image_embeddings.npy")

    # ─── TEXT EMBEDDINGS ──────────────────────────────────────────────────────────
    texts = [f"A photo of a handwritten digit {i}." for i in range(10)]
    tokens = tokenize(texts).to(device)

    with torch.no_grad():
        txt_embs = model.encode_text(tokens).cpu().numpy()
    np.save(os.path.join(output_dir, "mnist_text_embeddings.npy"), txt_embs)
    print(f"Saved text embeddings → {txt_embs.shape} to mnist_text_embeddings.npy")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Compute & save CLIP embeddings for MNIST images and digit prompts"
    )
    p.add_argument("--data-root",   default="./data",      help="MNIST data folder")
    p.add_argument("--batch-size",  type=int, default=64,   help="images per batch")
    p.add_argument("--model",       default="ViT-B/32",     help="CLIP model name")
    p.add_argument("--device",      default=None,           help="cpu or cuda")
    p.add_argument("--output-dir",  default="./embeddings", help="where to save .npy files")
    args = p.parse_args()

    compute_and_save_embeddings(
        data_root=args.data_root,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )