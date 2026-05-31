"""
CLIP large training on Modal — DistilBERT + ResNet-50 → 512-dim, on A10G.

Run:     modal run --detach train_large.py
Resume:  modal run --detach train_large.py --resume-from /checkpoints/large_epoch20.pt

Estimated cost: ~$5–6 for 40 epochs on Flickr30k (A10G @ $1.10/hr, ~5 hrs)

Checkpoints saved to the same 'clip-checkpoints' volume under the prefix 'large_'.
W&B run tagged with model=large so it sits alongside the small run in the same project.

NOTE: we actually change this to an A100
"""

import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-large", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = Path("/checkpoints")

CFG = dict(
    model_config="large",  # DistilBERT + ResNet-50 + 512-dim
    batch_size=256,
    epochs=40,
    lr=3e-4,  # slightly lower for larger model
    weight_decay=0.1,
    warmup_epochs=2,
    max_text_length=64,
    val_size=1000,
    log_matrix_every=100,
    save_every=5,
    num_workers=4,
    seed=42,
)


def similarity_matrix_figure(logits, step: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    mat = logits.detach().float().cpu().numpy()
    n = min(32, mat.shape[0])
    mat = mat[:n, :n]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Similarity matrix  (step {step})", fontsize=11)
    ax.set_xlabel("Text index")
    ax.set_ylabel("Image index")
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="lime", linewidth=1.2
            )
        )
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def recall_at_k(img_emb, txt_emb, k: int = 1) -> float:
    import torch

    sims = img_emb @ txt_emb.T
    ranks = sims.argsort(dim=1, descending=True)
    targets = torch.arange(len(img_emb), device=img_emb.device)
    hits = (ranks[:, :k] == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


@app.function(
    gpu="A100",  # ~1.5 hrs, ~$4.50 total
    timeout=60 * 60 * 8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(resume_from: str = ""):
    import random
    import numpy as np
    import torch
    import wandb
    from datasets import load_dataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import DistilBertTokenizerFast
    from tqdm import tqdm

    from model import CLIP, count_parameters
    from data import build_loaders

    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    wandb.init(
        project="clip-reproduction",
        config={**CFG, "model": "large"},
        tags=["large", "distilbert", "resnet50"],
        resume="allow",
    )

    # ---- Model -----------------------------------------------------------
    model = CLIP(config="large").to(device)
    params = count_parameters(model)
    print(f"Parameters: {params}")
    wandb.config.update(params)

    # ---- Data ------------------------------------------------------------
    print("Loading Flickr30k...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_loader, val_loader = build_loaders(
        ds,
        tokenizer,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        val_size=CFG["val_size"],
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CFG["epochs"]
    print(f"Train batches/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    # ---- Optimiser -------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    remaining_steps = (CFG["epochs"] - start_epoch) * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=1e-6)

    # ---- Training loop ---------------------------------------------------
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}"):
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(images, input_ids, attention_mask)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += loss.item()

            log = {
                "train/loss": loss.item(),
                "train/temperature": out["temperature"],
                "train/lr": scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if global_step % CFG["log_matrix_every"] == 0:
                log["train/similarity_matrix"] = similarity_matrix_figure(
                    out["logits"], global_step
                )
            wandb.log(log)
            global_step += 1

        # ---- Validation --------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_ie, all_te = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    out = model(images, input_ids, attention_mask)
                val_loss += out["loss"].item()
                all_ie.append(out["img_emb"])
                all_te.append(out["txt_emb"])

        avg_val = val_loss / len(val_loader)
        ie_all = torch.cat(all_ie)
        te_all = torch.cat(all_te)
        r1 = recall_at_k(ie_all, te_all, k=1)
        r5 = recall_at_k(ie_all, te_all, k=5)

        wandb.log(
            {
                "val/loss": avg_val,
                "val/R@1": r1,
                "val/R@5": r5,
                "val/similarity_matrix": similarity_matrix_figure(
                    (ie_all[:32] @ te_all[:32].T) * model.temperature, global_step
                ),
                "epoch": epoch + 1,
            }
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"train={epoch_loss/steps_per_epoch:.4f} | val={avg_val:.4f} | "
            f"R@1={r1:.3f} | R@5={r5:.3f} | τ={model.temperature:.3f} | "
            f"time={time.time()-t0:.0f}s"
        )

        # ---- Checkpoint --------------------------------------------------
        if (epoch + 1) % CFG["save_every"] == 0 or epoch == CFG["epochs"] - 1:
            path = CHECKPOINT_DIR / f"large_epoch{epoch+1:02d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": CFG,
                },
                path,
            )
            volume.commit()
            print(f"Saved → {path}")

    final = CHECKPOINT_DIR / "large_final.pt"
    torch.save({"model": model.state_dict(), "config": CFG}, final)
    volume.commit()
    print(f"Done. Final model → {final}")
    wandb.finish()


@app.local_entrypoint()
def main(resume_from: str = ""):
    train.remote(resume_from=resume_from)
