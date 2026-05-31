"""
Generate a t-SNE visualization of the CLIP embedding space.
Samples 1000 images, assigns rough semantic categories from captions,
and plots image + text embeddings colored by category.

Run:
    modal run generate_tsne.py
    # saves tsne_embeddings.png locally
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("scikit-learn")
    .add_local_python_source("model", "data")
)

app   = modal.App("clip-tsne", image=image)
volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=False)
CHECKPOINT_DIR = Path("/checkpoints")


@app.function(
    gpu="T4",
    timeout=60 * 20,
    volumes={CHECKPOINT_DIR: volume},
)
def make_tsne(n_samples: int = 1000) -> bytes:
    import random
    import io
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from datasets import load_dataset
    from transformers import DistilBertTokenizerFast
    from model import CLIP
    from data import VAL_TRANSFORM

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda")
    ckpt   = torch.load(CHECKPOINT_DIR / "large_final.pt", map_location=device)
    model  = CLIP(config="large").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # ── Load dataset ─────────────────────────────────────────────────────────
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]
    random.seed(42)
    indices = random.sample(range(len(ds)), n_samples)

    # ── Assign semantic categories from caption keywords ─────────────────────
    CATEGORIES = {
        "dog / animal":  ["dog", "puppy", "cat", "horse", "bird", "animal", "bear", "elephant"],
        "sport":         ["soccer", "football", "baseball", "basketball", "tennis",
                          "skiing", "swimming", "running", "player", "game", "sport"],
        "food / dining": ["eating", "food", "restaurant", "pizza", "cake", "drink",
                          "meal", "dining", "cook", "kitchen"],
        "water / beach": ["beach", "ocean", "water", "swimming", "surfing", "lake",
                          "river", "boat", "sea"],
        "children":      ["child", "children", "kid", "boy", "girl", "baby", "toddler"],
        "street / city": ["street", "city", "crowd", "market", "store", "building",
                          "urban", "road", "walk"],
        "nature":        ["mountain", "forest", "tree", "field", "grass", "park",
                          "flower", "outdoor", "garden"],
        "people":        ["man", "woman", "person", "people", "group", "friend"],
    }
    COLORS = {
        "dog / animal":  "#e74c3c",
        "sport":         "#3498db",
        "food / dining": "#f39c12",
        "water / beach": "#1abc9c",
        "children":      "#9b59b6",
        "street / city": "#e67e22",
        "nature":        "#27ae60",
        "people":        "#95a5a6",
    }

    def categorize(caption: str) -> str:
        cap = caption.lower()
        for cat, keywords in CATEGORIES.items():
            if any(k in cap for k in keywords):
                return cat
        return "people"   # default

    # ── Encode ───────────────────────────────────────────────────────────────
    all_ie, all_te, all_cats = [], [], []
    BATCH = 64
    for start in range(0, n_samples, BATCH):
        batch_idx = indices[start:start+BATCH]
        imgs, ids_list, masks = [], [], []
        for idx in batch_idx:
            row = ds[idx]
            cap = row["caption"][0]
            all_cats.append(categorize(cap))
            img = row["image"].convert("RGB")
            imgs.append(VAL_TRANSFORM(img))
            tok = tokenizer(cap, max_length=64, padding="max_length",
                            truncation=True, return_tensors="pt")
            ids_list.append(tok["input_ids"].squeeze(0))
            masks.append(tok["attention_mask"].squeeze(0))
        with torch.no_grad():
            all_ie.append(model.encode_image(torch.stack(imgs).to(device)).cpu())
            all_te.append(model.encode_text(
                torch.stack(ids_list).to(device),
                torch.stack(masks).to(device),
            ).cpu())

    ie = torch.cat(all_ie).numpy()   # (N, 512)
    te = torch.cat(all_te).numpy()   # (N, 512)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("Running t-SNE...")
    combined = np.concatenate([ie, te], axis=0)   # (2N, 512)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1200,
                random_state=42, learning_rate="auto", init="pca")
    proj = tsne.fit_transform(combined)
    img_2d = proj[:n_samples]
    txt_2d = proj[n_samples:]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    # Draw lines connecting matched pairs (every 5th for readability)
    for i in range(0, n_samples, 5):
        ax.plot([img_2d[i,0], txt_2d[i,0]],
                [img_2d[i,1], txt_2d[i,1]],
                color="#cccccc", lw=0.4, alpha=0.5, zorder=1)

    # Plot by category
    plotted = set()
    for i, cat in enumerate(all_cats):
        color = COLORS[cat]
        label_img = f"{cat} (image)" if (cat, "img") not in plotted else None
        label_txt = f"{cat} (text)"  if (cat, "txt") not in plotted else None

        ax.scatter(img_2d[i,0], img_2d[i,1], s=18, color=color,
                   alpha=0.75, zorder=3, marker="o",
                   edgecolors="white", linewidths=0.3)
        ax.scatter(txt_2d[i,0], txt_2d[i,1], s=18, color=color,
                   alpha=0.55, zorder=3, marker="D",
                   edgecolors="white", linewidths=0.3)
        plotted.add((cat, "img"))
        plotted.add((cat, "txt"))

    # Legend: one entry per category, circle=image diamond=text
    from matplotlib.lines import Line2D
    handles = []
    for cat, color in COLORS.items():
        handles.append(Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=color, markersize=8,
                              label=cat, alpha=0.85))
    handles.append(Line2D([0],[0], marker="o", color="w",
                          markerfacecolor="#555", markersize=7,
                          label="● image  ◆ text", alpha=0.7))

    ax.legend(handles=handles, fontsize=8.5, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#e5e5e3",
              title="Category", title_fontsize=9)

    ax.set_title("t-SNE of CLIP embedding space  ·  large model  ·  1,000 Flickr30k pairs",
                 fontsize=13, fontweight="600", color="#1a1a1a", pad=14)
    ax.set_xlabel("t-SNE dimension 1", fontsize=10, color="#6b6b6b")
    ax.set_ylabel("t-SNE dimension 2", fontsize=10, color="#6b6b6b")
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e5e3")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return buf.getvalue()


@app.local_entrypoint()
def main():
    print("Generating t-SNE (takes ~5 min)...")
    png = make_tsne.remote(n_samples=1000)
    out = Path("blog/tsne_embeddings.png")
    out.write_bytes(png)
    print(f"Saved → {out.resolve()}")
