import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from clip import load, tokenize  # assumes your local clip/ is on PYTHONPATH
from PIL import Image
from tqdm import tqdm

def compute_and_save_embeddings(
    data_root: str = "./data",
    batch_size: int = 64,
    model_name: str = "ViT-B/32",
    device: str = None,
    output_dir: str = "./embeddings",
    num_samples: int = 10
):
    print("Starting embedding computation...")
    # pick cpu or cuda
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # load CLIP model + preprocessor
    model, preprocess = load(model_name, device=device)
    model.eval()

    # ─── IMAGE EMBEDDINGS ─────────────────────────────────────────────────────────
    full_mnist_ds = MNIST(root=data_root, train=True, download=True,
                          transform=lambda img: preprocess(img.convert("RGB")))
    if num_samples is not None:
        mnist_ds = Subset(full_mnist_ds, range(num_samples))
    else:
        mnist_ds = full_mnist_ds
    loader = DataLoader(mnist_ds, batch_size=batch_size, shuffle=False)

    all_img_embs = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Encoding images"):
            imgs = imgs.to(device)
            embs = model.encode_image(imgs).cpu()
            all_img_embs.append(embs)
    all_img_embs = torch.cat(all_img_embs, dim=0).numpy()
    img_emb_path = os.path.join(output_dir, "mnist_image_embeddings.npy")
    np.save(img_emb_path, all_img_embs)
    print(f"Saved image embeddings → {all_img_embs.shape} to {img_emb_path}")

    # ─── TEXT EMBEDDINGS ──────────────────────────────────────────────────────────
    texts = [f"A photo of a handwritten digit {i}." for i in range(10)]
    tokens = tokenize(texts).to(device)

    with torch.no_grad():
        txt_embs = model.encode_text(tokens).cpu().numpy()
    text_emb_path = os.path.join(output_dir, "mnist_text_embeddings.npy")
    np.save(text_emb_path, txt_embs)
    print(f"Saved text embeddings → {txt_embs.shape} to {text_emb_path}")


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
    p.add_argument("--num-samples", type=int, default=None, help="number of MNIST samples to use")
    args = p.parse_args()

    compute_and_save_embeddings(
        data_root=args.data_root,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )