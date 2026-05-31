"""
CLIP evaluation on Modal — runs on T4 with access to the checkpoint volume.

Generates and saves to /checkpoints/figures/<model>/
  - similarity_matrix_16.png
  - similarity_matrix_32.png
  - recall_results.txt          (R@1, R@5, R@10)
  - failure_cases.png
  - query_<slug>.png            for each query

Run:
    modal run eval.py                          # small model (default)
    modal run eval.py --model large            # large model
    modal run eval.py --model large --queries "a dog in snow,sunset over ocean"
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-eval", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=False)
CHECKPOINT_DIR = Path("/checkpoints")

MODEL_CONFIGS = {
    "small": dict(checkpoint="clip_final.pt",  clip_config="small",
                  tokenizer="t5-small",        index="image_index.pt"),
    "large": dict(checkpoint="large_final.pt", clip_config="large",
                  tokenizer="distilbert",       index="image_index_large.pt"),
    "huge":  dict(checkpoint="huge_final.pt",  clip_config="huge",
                  tokenizer="minilm",           index="image_index_huge.pt"),
}

# Queries to run — override with --queries "a,b,c" from CLI
DEFAULT_QUERIES = [
    "a dog playing in the snow",
    "people eating at a restaurant",
    "a child riding a bicycle",
    "a sunset over the ocean",
    "a crowded city street at night",
    "two people hugging",
    "a cat sitting on a couch",
    "a soccer player kicking a ball",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_model(device, cfg):
    import torch
    from model import CLIP
    ckpt = torch.load(CHECKPOINT_DIR / cfg["checkpoint"], map_location=device)
    model = CLIP(config=cfg["clip_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded ({cfg['clip_config']})  |  τ = {model.temperature.item():.3f}")
    return model


def get_tokenizer(cfg):
    if cfg["tokenizer"] == "t5-small":
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained("t5-small")
    elif cfg["tokenizer"] == "minilm":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        from transformers import DistilBertTokenizerFast
        return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def build_or_load_index(model, device, cfg, batch_size=256):
    import torch
    from datasets import load_dataset
    from data import VAL_TRANSFORM

    index_path = CHECKPOINT_DIR / cfg["index"]
    if index_path.exists():
        print(f"Loading existing image index ({index_path.name})...")
        data = torch.load(index_path)
        return data["embeddings"], data["captions"]

    print("Building image index (one-time, ~3 min)...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]
    all_emb, all_captions = [], []

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        imgs = []
        for row in batch:
            img = row["image"].convert("RGB")
            imgs.append(VAL_TRANSFORM(img))
            all_captions.append(row["caption"][0])
        with torch.no_grad():
            emb = model.encode_image(torch.stack(imgs).to(device))
        all_emb.append(emb.cpu())
        if start % 2000 == 0:
            print(f"  {start}/{len(ds)}")

    emb_tensor = torch.cat(all_emb)
    torch.save({"embeddings": emb_tensor, "captions": all_captions}, index_path)
    volume.commit()
    print(f"Index saved: {emb_tensor.shape}")
    return emb_tensor, all_captions


def encode_text(model, tokenizer, text, device):
    import torch
    tokens = tokenizer(text, max_length=64, padding="max_length",
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        return model.encode_text(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )


# ---------------------------------------------------------------------------
# Figure: retrieval results for a single query
# ---------------------------------------------------------------------------

def save_query_figure(query, model, tokenizer, img_emb, captions, ds, device, top_k=6):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt_emb = encode_text(model, tokenizer, query, device)
    sims = (img_emb.to(device) @ txt_emb.T).squeeze(1).cpu()
    top_idx = sims.argsort(descending=True)[:top_k].tolist()
    top_scores = sims[top_idx].tolist()

    fig, axes = plt.subplots(1, top_k, figsize=(3.2 * top_k, 3.8))
    fig.suptitle(f'"{query}"', fontsize=13, fontweight="bold", y=1.03)

    for rank, (ax, idx, score) in enumerate(zip(axes, top_idx, top_scores)):
        img = ds[idx]["image"].convert("RGB")
        ax.imshow(img)
        ax.set_title(f"#{rank+1}  {score:.3f}", fontsize=9,
                     color="#27ae60" if rank == 0 else "#555")
        cap = captions[idx]
        ax.set_xlabel(cap[:55] + "…" if len(cap) > 55 else cap, fontsize=6.5, color="#444")
        ax.set_xticks([]); ax.set_yticks([])
        color = "#2ecc71" if rank == 0 else "#bbb"
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(3 if rank == 0 else 1)

    plt.tight_layout()
    slug = query.lower().replace(" ", "_")[:40]
    out = FIGURES_DIR / f"query_{slug}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure: similarity matrix
# ---------------------------------------------------------------------------

def save_similarity_matrix(model, tokenizer, ds, device, n=16, seed=42, tag=""):
    import torch
    import random
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from data import VAL_TRANSFORM

    random.seed(seed)
    indices = random.sample(range(len(ds)), n)

    imgs, ids_list, masks, row_captions = [], [], [], []
    for idx in indices:
        row = ds[idx]
        imgs.append(VAL_TRANSFORM(row["image"].convert("RGB")))
        cap = row["caption"][0]
        row_captions.append(cap[:28] + "…" if len(cap) > 28 else cap)
        tok = tokenizer(cap, max_length=64, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids_list.append(tok["input_ids"].squeeze(0))
        masks.append(tok["attention_mask"].squeeze(0))

    with torch.no_grad():
        ie = model.encode_image(torch.stack(imgs).to(device))
        te = model.encode_text(torch.stack(ids_list).to(device),
                               torch.stack(masks).to(device))
        sim = (ie @ te.T).cpu().numpy()

    diag = np.diag(sim)
    off_mask = ~np.eye(n, dtype=bool)
    off_mean = sim[off_mask].mean()
    print(f"[n={n}] pos mean={diag.mean():.3f}  neg mean={off_mean:.3f}  "
          f"separation={diag.mean()-off_mean:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    ax = axes[0]
    vabs = max(abs(sim.min()), abs(sim.max()))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    for i in range(n):
        ax.add_patch(patches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#00ff88", linewidth=2.2
        ))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"I{i}" for i in range(n)], fontsize=7)
    ax.set_xlabel("Text index", fontsize=11)
    ax.set_ylabel("Image index", fontsize=11)
    ax.set_title(f"Image–Text cosine similarity  (n={n})\nLime = positive pairs",
                 fontsize=12, fontweight="bold")

    # Distribution
    ax2 = axes[1]
    neg_scores = sim[off_mask]
    bins = np.linspace(sim.min() - 0.02, sim.max() + 0.02, 40)
    ax2.hist(neg_scores, bins=bins, color="#e74c3c", alpha=0.7,
             label=f"Negatives (n={len(neg_scores)})", density=True)
    ax2.hist(diag, bins=bins, color="#2ecc71", alpha=0.9,
             label=f"Positives (n={n})", density=True)
    ax2.axvline(diag.mean(), color="#27ae60", lw=2, ls="--",
                label=f"pos mean={diag.mean():.3f}")
    ax2.axvline(off_mean, color="#c0392b", lw=2, ls="--",
                label=f"neg mean={off_mean:.3f}")
    ax2.set_xlabel("Cosine similarity", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Score distribution:\nPositives vs Negatives",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"similarity_matrix{'_' + tag if tag else ''}.png"
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def compute_recalls(model, tokenizer, ds, device, n_val=1000, batch_size=128):
    import torch
    import random
    from data import VAL_TRANSFORM

    random.seed(42)
    indices = random.sample(range(len(ds)), n_val)
    all_ie, all_te = [], []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs, ids_list, masks = [], [], []
        for idx in batch_idx:
            row = ds[idx]
            imgs.append(VAL_TRANSFORM(row["image"].convert("RGB")))
            tok = tokenizer(row["caption"][0], max_length=64, padding="max_length",
                            truncation=True, return_tensors="pt")
            ids_list.append(tok["input_ids"].squeeze(0))
            masks.append(tok["attention_mask"].squeeze(0))
        with torch.no_grad():
            all_ie.append(model.encode_image(torch.stack(imgs).to(device)))
            all_te.append(model.encode_text(torch.stack(ids_list).to(device),
                                            torch.stack(masks).to(device)))

    ie = torch.cat(all_ie)
    te = torch.cat(all_te)
    sims = ie @ te.T
    targets = torch.arange(len(ie), device=ie.device)
    ranks = sims.argsort(dim=1, descending=True)

    lines = [f"Recall on {n_val} val images (image→text)\n" + "-" * 38]
    for k in [1, 5, 10]:
        hits = (ranks[:, :k] == targets.unsqueeze(1)).any(dim=1)
        r = hits.float().mean().item()
        lines.append(f"  R@{k:<3} = {r:.4f}  ({r*100:.1f}%)")
    result = "\n".join(lines)
    print(result)

    out = FIGURES_DIR / "recall_results.txt"
    out.write_text(result)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def save_failure_cases(model, tokenizer, img_emb, captions, ds, device,
                       n_failures=4, seed=0):
    import torch
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(seed)
    sample_idx = random.sample(range(len(ds)), 500)
    failures = []

    for idx in sample_idx:
        cap = ds[idx]["caption"][0]
        txt_emb = encode_text(model, tokenizer, cap, device)
        sims = (img_emb.to(device) @ txt_emb.T).squeeze(1).cpu()
        top1 = sims.argmax().item()
        if top1 != idx:
            failures.append((idx, top1, sims[top1].item(), cap))
        if len(failures) >= n_failures:
            break

    print(f"Found {len(failures)} failure cases")
    fig, axes = plt.subplots(n_failures, 2, figsize=(7, 3.5 * n_failures))
    fig.suptitle("Failure cases — retrieved vs ground-truth",
                 fontsize=13, fontweight="bold")

    for row_i, (true_idx, pred_idx, score, cap) in enumerate(failures):
        for col_i, (img_idx, title, color) in enumerate([
            (pred_idx, f"Retrieved  (sim={score:.3f})", "#e74c3c"),
            (true_idx, "Ground truth", "#2ecc71"),
        ]):
            ax = axes[row_i][col_i]
            ax.imshow(ds[img_idx]["image"].convert("RGB"))
            ax.set_title(title, fontsize=9, color=color, fontweight="bold")
            if col_i == 0:
                label = f'"{cap[:42]}…"' if len(cap) > 42 else f'"{cap}"'
                ax.set_ylabel(label, fontsize=7.5, color="#333", labelpad=6)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(2)

    plt.tight_layout()
    out = FIGURES_DIR / "failure_cases.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60 * 30,
    volumes={CHECKPOINT_DIR: volume},
)
def run_eval(queries: list[str], model_name: str = "small"):
    import torch
    from datasets import load_dataset

    cfg = MODEL_CONFIGS[model_name]
    # Save figures in a per-model subfolder so small and large don't overwrite each other
    figures_dir = Path(f"/checkpoints/figures/{model_name}")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the module-level FIGURES_DIR used by all helpers
    import eval as _self
    _self.FIGURES_DIR = figures_dir

    device = torch.device("cuda")

    model    = load_model(device, cfg)
    tokenizer = get_tokenizer(cfg)

    print("Loading Flickr30k...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    img_emb, captions = build_or_load_index(model, device, cfg)

    print("\n--- Similarity matrices ---")
    save_similarity_matrix(model, tokenizer, ds, device, n=16, seed=42, tag="16")
    save_similarity_matrix(model, tokenizer, ds, device, n=32, seed=7,  tag="32")

    print("\n--- Recall@k ---")
    compute_recalls(model, tokenizer, ds, device)

    print("\n--- Query figures ---")
    for q in queries:
        save_query_figure(q, model, tokenizer, img_emb, captions, ds, device)

    print("\n--- Failure cases ---")
    save_failure_cases(model, tokenizer, img_emb, captions, ds, device)

    volume.commit()
    print(f"\nAll figures saved to {figures_dir}")


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(queries: str = "", model: str = "small"):
    if model not in MODEL_CONFIGS:
        print(f"Unknown model '{model}'. Choose: {list(MODEL_CONFIGS.keys())}")
        return
    q_list = [q.strip() for q in queries.split(",")] if queries else DEFAULT_QUERIES
    print(f"Running eval for model='{model}'")
    run_eval.remote(queries=q_list, model_name=model)
