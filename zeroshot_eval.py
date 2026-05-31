"""
Zero-shot classification evaluation for CLIP reproductions.
Tests on CIFAR-100 and Food-101 using prompt-based text embeddings.

How it works:
  1. Encode each class name as "a photo of a {class}" → text embeddings
  2. Encode all test images → image embeddings
  3. For each image, pick the class whose text embedding is nearest
  4. Compute top-1 and top-5 accuracy

Run:
    modal run zeroshot_eval.py                  # both models, both datasets
    modal run zeroshot_eval.py --model large
    modal run zeroshot_eval.py --dataset cifar100
    modal run zeroshot_eval.py --model large --dataset food101
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-zeroshot", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=False)
CHECKPOINT_DIR = Path("/checkpoints")

MODEL_CONFIGS = {
    "small": dict(checkpoint="clip_final.pt",  clip_config="small", tokenizer="t5-small"),
    "large": dict(checkpoint="large_final.pt", clip_config="large", tokenizer="distilbert"),
}

# Prompt templates — average over multiple prompts for better accuracy
# (technique from the original CLIP paper §3.3)
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of many {}.",
    "a photo of the {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_cifar100():
    """Returns (images_pil, labels_int, classnames)."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root="/tmp/cifar100", train=False, download=True)
    classnames = ds.classes  # list of 100 string names
    images = [ds[i][0] for i in range(len(ds))]   # PIL images
    labels = [ds[i][1] for i in range(len(ds))]
    return images, labels, classnames


def load_food101():
    """Returns (images_pil, labels_int, classnames)."""
    from torchvision.datasets import Food101
    ds = Food101(root="/tmp/food101", split="test", download=True)
    classnames = [c.replace("_", " ") for c in ds.classes]
    images = [ds[i][0] for i in range(len(ds))]   # PIL images
    labels = [ds[i][1] for i in range(len(ds))]
    return images, labels, classnames


DATASETS = {
    "cifar100": load_cifar100,
    "food101":  load_food101,
}


# ---------------------------------------------------------------------------
# Core zero-shot eval
# ---------------------------------------------------------------------------

def build_text_embeddings(model, tokenizer, classnames, device):
    """
    For each class, encode all prompt templates and average the embeddings.
    Returns (n_classes, embed_dim) normalised tensor.
    """
    import torch
    import torch.nn.functional as F

    all_class_emb = []
    for cls in classnames:
        prompts = [t.format(cls) for t in PROMPT_TEMPLATES]
        tokens = tokenizer(
            prompts,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = model.encode_text(
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device),
            )  # (n_templates, D)
        # Average over templates then renormalise
        mean_emb = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
        all_class_emb.append(mean_emb)

    return torch.cat(all_class_emb)  # (n_classes, D)


def encode_images(model, images_pil, device, batch_size=256):
    import torch
    from data import VAL_TRANSFORM

    all_emb = []
    for start in range(0, len(images_pil), batch_size):
        batch = images_pil[start:start + batch_size]
        tensors = []
        for img in batch:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(VAL_TRANSFORM(img))
        with torch.no_grad():
            emb = model.encode_image(torch.stack(tensors).to(device))
        all_emb.append(emb.cpu())
        if start % 2000 == 0 and start > 0:
            print(f"    encoded {start}/{len(images_pil)}")
    return torch.cat(all_emb)  # (N, D)


def compute_accuracy(img_emb, text_emb, labels, device, batch_size=512):
    """Returns top-1 and top-5 accuracy."""
    import torch
    text_emb = text_emb.to(device)
    correct_1 = correct_5 = 0

    for start in range(0, len(img_emb), batch_size):
        batch = img_emb[start:start + batch_size].to(device)
        sims = batch @ text_emb.T          # (B, n_classes)
        batch_labels = torch.tensor(
            labels[start:start + batch_size], device=device
        )
        # Top-1
        preds_1 = sims.argmax(dim=1)
        correct_1 += (preds_1 == batch_labels).sum().item()
        # Top-5
        top5 = sims.topk(5, dim=1).indices
        correct_5 += (top5 == batch_labels.unsqueeze(1)).any(dim=1).sum().item()

    n = len(img_emb)
    return correct_1 / n, correct_5 / n


def per_class_accuracy(img_emb, text_emb, labels, classnames, device):
    """Returns dict of {classname: top1_accuracy}."""
    import torch
    text_emb = text_emb.to(device)
    preds = (img_emb.to(device) @ text_emb.T).argmax(dim=1).cpu().tolist()
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)
    for pred, label in zip(preds, labels):
        total[label] += 1
        if pred == label:
            correct[label] += 1
    return {classnames[i]: correct[i] / total[i] for i in range(len(classnames))}


# ---------------------------------------------------------------------------
# Figure: success and failure examples
# ---------------------------------------------------------------------------

def make_examples_figure(images_pil, labels, classnames, img_emb, text_emb,
                         device, n_each=8, seed=42):
    """
    Shows n_each successes and n_each failures.
    Each tile: image + true label (green/red) + top-3 predicted classes with scores.
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import random
    import io

    random.seed(seed)
    text_emb = text_emb.to(device)

    # Get all predictions and scores in one pass
    sims = (img_emb.to(device) @ text_emb.T).cpu()          # (N, n_classes)
    top3_scores, top3_preds = sims.topk(3, dim=1)            # (N, 3) each
    top1_preds = top3_preds[:, 0].tolist()

    successes, failures = [], []
    indices = list(range(len(labels)))
    random.shuffle(indices)

    for i in indices:
        true = labels[i]
        pred = top1_preds[i]
        entry = dict(
            idx=i,
            true_label=classnames[true],
            pred_label=classnames[pred],
            correct=(pred == true),
            top3=[(classnames[top3_preds[i, k].item()],
                   top3_scores[i, k].item()) for k in range(3)],
        )
        if pred == true and len(successes) < n_each:
            successes.append(entry)
        elif pred != true and len(failures) < n_each:
            failures.append(entry)
        if len(successes) >= n_each and len(failures) >= n_each:
            break

    n_cols = n_each
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 7.5))
    fig.patch.set_facecolor("white")

    section_labels = [("✓  Correct predictions", "#15803d"),
                      ("✗  Incorrect predictions", "#b91c1c")]

    for row, (examples, (section_title, section_color)) in enumerate(
        zip([successes, failures], section_labels)
    ):
        # Section label on the left
        fig.text(0.01, 0.74 - row * 0.48, section_title,
                 fontsize=11, fontweight="600", color=section_color,
                 va="center", rotation=90)

        for col, ex in enumerate(examples):
            ax = axes[row][col]
            img = images_pil[ex["idx"]]
            if img.mode != "RGB":
                img = img.convert("RGB")
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])

            # Coloured border
            border_color = "#16a34a" if ex["correct"] else "#dc2626"
            for s in ax.spines.values():
                s.set_edgecolor(border_color); s.set_linewidth(2.5)

            # True label at top
            ax.set_title(ex["true_label"], fontsize=8, fontweight="600",
                         color="#1a1a1a", pad=4)

            # Top-3 predictions below image
            lines = []
            for rank, (cls, score) in enumerate(ex["top3"]):
                marker = "→ " if rank == 0 else f"{rank+1}. "
                lines.append(f"{marker}{cls[:18]:<18} {score:.2f}")
            caption = "\n".join(lines)

            ax.set_xlabel(caption, fontsize=6.5, color="#444",
                          fontfamily="monospace", labelpad=4)

    plt.suptitle("", y=0)  # spacer
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Figure: overall bar + per-class breakdown
# ---------------------------------------------------------------------------

def make_results_figure(dataset_name, model_name, top1, top5,
                        per_class, classnames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    accs = [per_class[c] for c in classnames]
    sorted_idx = np.argsort(accs)
    n = len(classnames)

    # Show top-20 and bottom-20 classes
    show_idx = list(sorted_idx[:20]) + list(sorted_idx[-20:])
    show_names = [classnames[i] for i in show_idx]
    show_accs  = [accs[i]       for i in show_idx]
    colors = ["#dc2626"] * 20 + ["#16a34a"] * 20

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1, 2.5]})
    fig.suptitle(
        f"Zero-shot classification — {dataset_name}  ·  model: {model_name}",
        fontsize=13, fontweight="600", y=1.01
    )

    # ---- Left: summary card ----
    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.72, f"{top1*100:.1f}%", ha="center", va="center",
            fontsize=52, fontweight="700", color="#1a1a1a",
            transform=ax.transAxes)
    ax.text(0.5, 0.55, "Top-1 Accuracy", ha="center", va="center",
            fontsize=13, color="#6b6b6b", transform=ax.transAxes)
    ax.text(0.5, 0.38, f"{top5*100:.1f}%", ha="center", va="center",
            fontsize=32, fontweight="600", color="#6b6b6b",
            transform=ax.transAxes)
    ax.text(0.5, 0.26, "Top-5 Accuracy", ha="center", va="center",
            fontsize=11, color="#aaa", transform=ax.transAxes)
    ax.text(0.5, 0.10,
            f"{n} classes  ·  zero-shot\nno fine-tuning",
            ha="center", va="center", fontsize=9,
            color="#aaa", transform=ax.transAxes, style="italic")

    # border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#e5e5e3")
        spine.set_linewidth(1)

    # ---- Right: per-class bars (bottom 20 + top 20) ----
    ax2 = axes[1]
    y = np.arange(40)
    bars = ax2.barh(y, show_accs, color=colors, height=0.7, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(show_names, fontsize=8)
    ax2.set_xlabel("Top-1 accuracy per class", fontsize=10)
    ax2.axvline(top1, color="#1a1a1a", lw=1.2, ls="--",
                label=f"overall mean {top1*100:.1f}%")
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(fontsize=9, loc="lower right")

    # Separator between worst/best groups
    ax2.axhline(19.5, color="#e5e5e3", lw=1.5, ls="-")
    ax2.text(1.02, 9.5,  "worst\n20", fontsize=8, color="#dc2626",
             va="center", ha="left")
    ax2.text(1.02, 29.5, "best\n20",  fontsize=8, color="#16a34a",
             va="center", ha="left")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#e5e5e3")
    ax2.spines["bottom"].set_color("#e5e5e3")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    return buf.getvalue()


def make_comparison_figure(results: dict):
    """
    results = {
      (model, dataset): {"top1": float, "top5": float}
    }
    Bar chart comparing all combinations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    models   = ["small", "large"]
    datasets = ["cifar100", "food101"]
    colors   = {"small": "#6b7280", "large": "#1a1a1a"}
    x = np.arange(len(datasets))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    fig.suptitle("Zero-shot classification: small vs large model",
                 fontsize=13, fontweight="600", y=1.02)

    for ax, metric, label in zip(axes, ["top1", "top5"], ["Top-1", "Top-5"]):
        for i, model in enumerate(models):
            vals = [results.get((model, ds), {}).get(metric, 0) * 100
                    for ds in datasets]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=model, color=colors[model],
                          alpha=0.85, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="500")

        ax.set_xticks(x)
        ax.set_xticklabels(["CIFAR-100", "Food-101"], fontsize=11)
        ax.set_ylabel(f"{label} accuracy (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="600")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_ylim(0, max(
            results.get((m, d), {}).get(metric, 0) * 100
            for m in models for d in datasets
        ) * 1.2 + 5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e5e3")
        ax.spines["bottom"].set_color("#e5e5e3")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60 * 60 * 2,
    volumes={CHECKPOINT_DIR: volume},
)
def eval_zeroshot(model_name: str, dataset_name: str) -> dict:
    import torch

    device = torch.device("cuda")
    cfg = MODEL_CONFIGS[model_name]

    # Load model
    from model import CLIP
    ckpt = torch.load(CHECKPOINT_DIR / cfg["checkpoint"], map_location=device)
    model = CLIP(config=cfg["clip_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {model_name} model  |  τ={model.temperature.item():.3f}")

    # Load tokenizer
    if cfg["tokenizer"] == "t5-small":
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    else:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Load dataset
    print(f"Loading {dataset_name}...")
    images, labels, classnames = DATASETS[dataset_name]()
    print(f"  {len(images)} images, {len(classnames)} classes")

    # Build text embeddings (one per class, averaged over prompt templates)
    print("Building text embeddings...")
    text_emb = build_text_embeddings(model, tokenizer, classnames, device)
    print(f"  text_emb: {text_emb.shape}")

    # Encode all test images
    print("Encoding images...")
    img_emb = encode_images(model, images, device)
    print(f"  img_emb: {img_emb.shape}")

    # Compute accuracy
    print("Computing accuracy...")
    top1, top5 = compute_accuracy(img_emb, text_emb, labels, device)
    print(f"  Top-1: {top1*100:.2f}%  |  Top-5: {top5*100:.2f}%")

    # Per-class breakdown
    per_class = per_class_accuracy(img_emb, text_emb, labels, classnames, device)

    # Figures
    results_fig  = make_results_figure(
        dataset_name, model_name, top1, top5, per_class, classnames
    )
    examples_fig = make_examples_figure(
        images, labels, classnames, img_emb, text_emb, device, n_each=8
    )

    return {
        "top1": top1,
        "top5": top5,
        "per_class": per_class,
        "figure": results_fig,
        "examples_figure": examples_fig,
        "model": model_name,
        "dataset": dataset_name,
    }


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(model: str = "both", dataset: str = "both"):
    models   = ["small", "large"] if model   == "both" else [model]
    datasets = ["cifar100", "food101"] if dataset == "both" else [dataset]

    out_dir = Path("results/zeroshot")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Launch all combinations in parallel
    import modal as _modal
    futures = {}
    for m in models:
        for d in datasets:
            print(f"Submitting: {m} × {d}")
            futures[(m, d)] = eval_zeroshot.spawn(
                model_name=m, dataset_name=d
            )

    # Collect results
    all_results = {}
    txt_lines = []
    for (m, d), future in futures.items():
        print(f"Waiting for {m} × {d}...")
        res = future.get()
        all_results[(m, d)] = {"top1": res["top1"], "top5": res["top5"]}

        # Save figures
        fname = f"{m}_{d}.png"
        (out_dir / fname).write_bytes(res["figure"])
        print(f"  Saved → results/zeroshot/{fname}")

        ex_fname = f"{m}_{d}_examples.png"
        (out_dir / ex_fname).write_bytes(res["examples_figure"])
        print(f"  Saved → results/zeroshot/{ex_fname}")

        txt_lines.append(
            f"{m:8s}  {d:12s}  Top-1={res['top1']*100:.2f}%  "
            f"Top-5={res['top5']*100:.2f}%"
        )

        # Print top-5 and bottom-5 classes
        pc = res["per_class"]
        sorted_cls = sorted(pc.items(), key=lambda x: x[1])
        print(f"\n  Worst 5 classes ({m}/{d}):")
        for cls, acc in sorted_cls[:5]:
            print(f"    {cls:<30s} {acc*100:.1f}%")
        print(f"  Best 5 classes ({m}/{d}):")
        for cls, acc in sorted_cls[-5:]:
            print(f"    {cls:<30s} {acc*100:.1f}%")

    # Summary text file
    summary = "\n".join([
        "Zero-shot classification results",
        "=" * 50,
        f"{'Model':<8}  {'Dataset':<12}  Top-1       Top-5",
        "-" * 50,
    ] + txt_lines)
    (out_dir / "summary.txt").write_text(summary)
    print(f"\n{summary}")

    # Comparison figure (only if we have both models + both datasets)
    if len(all_results) > 1:
        comp_bytes = make_comparison_figure(all_results)
        (out_dir / "comparison.png").write_bytes(comp_bytes)
        print(f"Saved → results/zeroshot/comparison.png")
