"""
Run all demo queries for both small and large models and save to:
  results/small/  — text→image and image→text from the small model
  results/large/  — same for the large model

Usage:
    modal run run_examples.py              # both models
    modal run run_examples.py --model small
    modal run run_examples.py --model large
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-examples", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=False)
CHECKPOINT_DIR = Path("/checkpoints")

# ── Queries to run text→image ────────────────────────────────────────────────
TEXT_QUERIES = [
    "a dog playing in the snow",
    "people eating at a restaurant",
    "a child riding a bicycle",
    "a sunset over the ocean",
    "a crowded city street at night",
    "two people hugging",
    "a cat sitting on a couch",
    "a soccer player kicking a ball",
]

# ── Example images to run image→text ────────────────────────────────────────
EXAMPLE_IMAGES = ["ex_1.jpg", "ex_2.jpg", "ex_3.jpg"]

MODEL_CONFIGS = {
    "small": dict(
        checkpoint="clip_final.pt",
        clip_config="small",
        tokenizer="t5-small",
        index="image_index.pt",
    ),
    "large": dict(
        checkpoint="large_final.pt",
        clip_config="large",
        tokenizer="distilbert",
        index="image_index_large.pt",
    ),
    "huge": dict(
        checkpoint="huge_final.pt",
        clip_config="huge",
        tokenizer="minilm",
        index="image_index_huge.pt",
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(device, cfg):
    import torch
    from model import CLIP
    ckpt = torch.load(CHECKPOINT_DIR / cfg["checkpoint"], map_location=device)
    model = CLIP(config=cfg["clip_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded {cfg['clip_config']} model  |  τ={model.temperature.item():.3f}")
    return model


def get_tokenizer(cfg):
    if cfg["tokenizer"] == "t5-small":
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained("t5-small")
    elif cfg["tokenizer"] == "minilm":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    from transformers import DistilBertTokenizerFast
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def build_or_load_index(model, device, cfg, batch_size=256):
    import torch
    from datasets import load_dataset
    from data import VAL_TRANSFORM

    index_path = CHECKPOINT_DIR / cfg["index"]
    if index_path.exists():
        print(f"  Loading image index ({index_path.name})...")
        d = torch.load(index_path)
        return d["embeddings"], d["captions"]

    print("  Building image index (one-time ~3 min)...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]
    all_emb, all_captions = [], []
    for start in range(0, len(ds), batch_size):
        rows = ds.select(range(start, min(start + batch_size, len(ds))))
        imgs = []
        for row in rows:
            imgs.append(VAL_TRANSFORM(row["image"].convert("RGB")))
            all_captions.append(row["caption"][0])
        with torch.no_grad():
            emb = model.encode_image(torch.stack(imgs).to(device))
        all_emb.append(emb.cpu())
    emb_tensor = torch.cat(all_emb)
    torch.save({"embeddings": emb_tensor, "captions": all_captions}, index_path)
    volume.commit()
    return emb_tensor, all_captions


def render_text_to_image(query, top_idx, top_scores, captions, ds, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    top_k = len(top_idx)
    fig, axes = plt.subplots(1, top_k, figsize=(3.2 * top_k, 4.2))
    fig.suptitle(f'"{query}"',
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.01, f"Model: {model_name}  ·  Text → Image",
             ha="center", fontsize=9, color="#888", style="italic")

    for rank, (ax, idx, score) in enumerate(zip(axes, top_idx, top_scores)):
        img = ds[idx]["image"].convert("RGB")
        ax.imshow(img)
        ax.set_title(f"#{rank+1}   {score:.3f}", fontsize=9,
                     color="#16a34a" if rank == 0 else "#555")
        cap = captions[idx]
        ax.set_xlabel(cap[:52] + "…" if len(cap) > 52 else cap,
                      fontsize=7, color="#555")
        ax.set_xticks([]); ax.set_yticks([])
        color = "#16a34a" if rank == 0 else "#d1d5db"
        lw = 2.5 if rank == 0 else 1
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(lw)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def render_image_to_text(query_img_pil, top_idx, top_scores,
                          cap_texts, cap_img_indices, ds, model_name, img_name):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    top_k = len(top_idx)
    fig = plt.figure(figsize=(3.5 + 3.2 * top_k, 8))
    fig.suptitle(f"Image → Text  ·  {img_name}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01, f"Model: {model_name}",
             ha="center", fontsize=9, color="#888", style="italic")

    gs = fig.add_gridspec(2, 1 + top_k,
                          height_ratios=[3, 1.4],
                          hspace=0.08, wspace=0.12)

    # Query image
    ax_q = fig.add_subplot(gs[:, 0])
    ax_q.imshow(query_img_pil)
    ax_q.set_title("Query\nimage", fontsize=10, fontweight="bold", pad=6)
    ax_q.set_xticks([]); ax_q.set_yticks([])
    for s in ax_q.spines.values():
        s.set_edgecolor("#2563eb"); s.set_linewidth(2.5)

    for rank, (cap_idx, score) in enumerate(zip(top_idx, top_scores)):
        col = rank + 1
        caption_text = cap_texts[cap_idx]
        src_idx = cap_img_indices[cap_idx]
        color = "#16a34a" if rank == 0 else "#aaa"

        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(ds[src_idx]["image"].convert("RGB"))
        ax_img.set_title(f"#{rank+1}   sim={score:.3f}", fontsize=9,
                         color="#16a34a" if rank == 0 else "#666", pad=5)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for s in ax_img.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2.5 if rank == 0 else 1)

        ax_cap = fig.add_subplot(gs[1, col])
        ax_cap.axis("off")
        wrapped = "\n".join(textwrap.wrap(caption_text, width=32))
        ax_cap.text(0.5, 0.95, wrapped, fontsize=8, va="top", ha="center",
                    transform=ax_cap.transAxes, linespacing=1.5,
                    color="#111", style="italic",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#f9fafb",
                              edgecolor=color, linewidth=1.2))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


# ── Modal function ────────────────────────────────────────────────────────────

@app.function(
    gpu="T4",
    timeout=60 * 60,
    volumes={CHECKPOINT_DIR: volume},
)
def run_all(model_name: str, image_bytes_map: dict) -> dict:
    """
    Returns a dict of { filename: png_bytes } for the caller to write locally.
    """
    import torch
    from datasets import load_dataset
    from data import VAL_TRANSFORM
    from PIL import Image
    import io

    device = torch.device("cuda")
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n=== Model: {model_name} ===")

    model    = load_model(device, cfg)
    tokenizer = get_tokenizer(cfg)

    print("  Loading Flickr30k...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]
    img_emb, captions = build_or_load_index(model, device, cfg)

    results = {}

    # ── Text → Image ────────────────────────────────────────────────────────
    print(f"\n  Text→Image ({len(TEXT_QUERIES)} queries)...")
    for query in TEXT_QUERIES:
        tokens = tokenizer(query, max_length=64, padding="max_length",
                           truncation=True, return_tensors="pt")
        with torch.no_grad():
            txt_emb = model.encode_text(
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device),
            )
        sims = (img_emb.to(device) @ txt_emb.T).squeeze(1).cpu()
        top_idx    = sims.argsort(descending=True)[:6].tolist()
        top_scores = sims[top_idx].tolist()

        png = render_text_to_image(query, top_idx, top_scores,
                                   captions, ds, model_name)
        slug = query.lower().replace(" ", "_")[:40]
        results[f"t2i_{slug}.png"] = png
        print(f"    ✓ {query[:45]}")

    # ── Image → Text ────────────────────────────────────────────────────────
    # Build caption index
    cap_index_path = CHECKPOINT_DIR / f"caption_index_{model_name}.pt"
    if cap_index_path.exists():
        print(f"\n  Loading caption index...")
        d = torch.load(cap_index_path)
        cap_emb, cap_texts, cap_img_indices = (
            d["embeddings"], d["captions"], d["image_indices"]
        )
    else:
        print(f"\n  Building caption index (~2 min)...")
        cap_emb_list, cap_texts, cap_img_indices = [], [], []
        batch_ids, batch_masks = [], []

        def flush(start_img):
            nonlocal batch_ids, batch_masks
            if not batch_ids:
                return
            with torch.no_grad():
                e = model.encode_text(
                    torch.stack(batch_ids).to(device),
                    torch.stack(batch_masks).to(device),
                )
            cap_emb_list.append(e.cpu())
            batch_ids, batch_masks = [], []

        for img_i in range(len(ds)):
            for cap in ds[img_i]["caption"]:
                tok = tokenizer(cap, max_length=64, padding="max_length",
                                truncation=True, return_tensors="pt")
                batch_ids.append(tok["input_ids"].squeeze(0))
                batch_masks.append(tok["attention_mask"].squeeze(0))
                cap_texts.append(cap)
                cap_img_indices.append(img_i)
                if len(batch_ids) == 256:
                    flush(img_i)
            if img_i % 5000 == 0:
                print(f"    {img_i}/{len(ds)}")
        flush(len(ds))

        cap_emb = torch.cat(cap_emb_list)
        torch.save({"embeddings": cap_emb,
                    "captions": cap_texts,
                    "image_indices": cap_img_indices}, cap_index_path)
        volume.commit()
        print(f"  Caption index saved: {cap_emb.shape}")

    print(f"\n  Image→Text ({len(image_bytes_map)} images)...")
    for img_name, img_bytes in image_bytes_map.items():
        query_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = VAL_TRANSFORM(query_img).unsqueeze(0).to(device)
        with torch.no_grad():
            qemb = model.encode_image(img_tensor)

        sims = (cap_emb.to(device) @ qemb.T).squeeze(1).cpu()
        top_idx    = sims.argsort(descending=True)[:6].tolist()
        top_scores = sims[top_idx].tolist()

        png = render_image_to_text(query_img, top_idx, top_scores,
                                   cap_texts, cap_img_indices,
                                   ds, model_name, img_name)
        stem = Path(img_name).stem
        results[f"i2t_{stem}.png"] = png
        print(f"    ✓ {img_name}")

    return results


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "both"):
    models = ["small", "large"] if model == "both" else [model]

    # Load example images locally once
    ex_dir = Path("/Users/ethanhersch/Desktop/CLIP_demo/example_images")
    image_bytes_map = {
        p.name: p.read_bytes()
        for p in sorted(ex_dir.glob("*.jpg")) + sorted(ex_dir.glob("*.png"))
    }
    print(f"Loaded {len(image_bytes_map)} example images: {list(image_bytes_map.keys())}")

    for model_name in models:
        print(f"\nRunning model: {model_name}")
        results = run_all.remote(
            model_name=model_name,
            image_bytes_map=image_bytes_map,
        )

        out_dir = Path(f"results/{model_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname, png_bytes in results.items():
            (out_dir / fname).write_bytes(png_bytes)
            print(f"  Saved → results/{model_name}/{fname}")

        print(f"  Done. {len(results)} files in results/{model_name}/")
