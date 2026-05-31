"""
Interactive CLIP retrieval demo.

Usage:
    # First time: build the image index (runs once, saves index.pt)
    python demo.py --checkpoint clip_final.pt --build-index

    # Then query interactively
    python demo.py --checkpoint clip_final.pt --query "a dog playing in snow"

    # Show top-k (default 5)
    python demo.py --checkpoint clip_final.pt --query "sunset over the ocean" --top-k 8
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from transformers import T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from model import CLIP
from data import VAL_TRANSFORM


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_index(model, tokenizer, device, index_path: Path, batch_size: int = 64):
    """Encode all Flickr30k images and save embeddings + metadata."""
    print("Loading Flickr30k …")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    model.eval()
    all_emb, all_captions, all_images_pil = [], [], []

    for start in tqdm(range(0, len(ds), batch_size), desc="Encoding images"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        imgs = []
        for row in batch:
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(VAL_TRANSFORM(img))
            all_captions.append(row["caption"][0])  # store first caption as label
            all_images_pil.append(img)

        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        all_emb.append(emb.cpu())

    all_emb = torch.cat(all_emb)  # (N, 256)

    torch.save({
        "embeddings": all_emb,
        "captions": all_captions,
        "image_count": len(all_captions),
    }, index_path)
    print(f"Index built: {len(all_captions)} images → {index_path}")
    return all_emb, all_captions, all_images_pil


def load_index(index_path: Path):
    data = torch.load(index_path, map_location="cpu")
    return data["embeddings"], data["captions"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    model: CLIP,
    tokenizer: T5Tokenizer,
    img_embeddings: torch.Tensor,
    captions: list,
    device,
    top_k: int = 5,
):
    model.eval()
    tokens = tokenizer(
        query, max_length=64, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        txt_emb = model.encode_text(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )  # (1, 256)

    sims = (img_embeddings.to(device) @ txt_emb.T).squeeze(1)  # (N,)
    top_indices = sims.argsort(descending=True)[:top_k].cpu().tolist()
    top_scores = sims[top_indices].cpu().tolist()
    return top_indices, top_scores


# ---------------------------------------------------------------------------
# Visualisation (blog-quality figures)
# ---------------------------------------------------------------------------

def show_results(
    query: str,
    indices: list,
    scores: list,
    captions: list,
    ds,
    save_path: Path = None,
):
    k = len(indices)
    fig = plt.figure(figsize=(4 * k, 5))
    fig.suptitle(f'Query: "{query}"', fontsize=14, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, k, figure=fig, wspace=0.05)

    for rank, (idx, score) in enumerate(zip(indices, scores)):
        ax = fig.add_subplot(gs[0, rank])
        img = ds[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        ax.imshow(img)
        ax.set_title(f"#{rank+1}  sim={score:.3f}", fontsize=9)
        # Wrap caption
        cap = captions[idx]
        cap_short = cap[:60] + "…" if len(cap) > 60 else cap
        ax.set_xlabel(cap_short, fontsize=7, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ecc71" if rank == 0 else "#cccccc")
            spine.set_linewidth(2.5 if rank == 0 else 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CLIP text→image retrieval demo")
    p.add_argument("--checkpoint", required=True, help="Path to clip_final.pt")
    p.add_argument("--index", default="image_index.pt", help="Path to save/load image index")
    p.add_argument("--build-index", action="store_true", help="(Re)build the image index")
    p.add_argument("--query", type=str, default=None, help="Text query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--save-fig", type=str, default=None, help="Save result figure to this path")
    p.add_argument("--interactive", action="store_true", help="Loop for multiple queries")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {"embed_dim": 256})
    model = CLIP(embed_dim=cfg["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded.")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    index_path = Path(args.index)

    # Build or load index
    if args.build_index or not index_path.exists():
        ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]
        img_emb, captions = build_index(model, tokenizer, device, index_path)[:2]
    else:
        img_emb, captions = load_index(index_path)
        ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    def run_query(query: str):
        indices, scores = retrieve(query, model, tokenizer, img_emb, captions, device, args.top_k)
        save = Path(args.save_fig) if args.save_fig else None
        show_results(query, indices, scores, captions, ds, save_path=save)

    if args.interactive:
        print("\nEnter text queries (Ctrl+C to quit):")
        while True:
            try:
                q = input("\n> ").strip()
                if q:
                    run_query(q)
            except KeyboardInterrupt:
                print("\nBye!")
                break
    elif args.query:
        run_query(args.query)
    else:
        print("Pass --query 'some text' or --interactive to start querying.")


if __name__ == "__main__":
    main()
