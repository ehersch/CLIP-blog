"""
CLIP training on Modal (single T4 GPU).

Run locally:  modal run train.py
Resume:       modal run train.py --resume-from /path/to/checkpoint.pt
Estimated cost: ~$1.50–2.50 for 20 epochs on Flickr30k (T4 @ $0.76/hr)
"""

import math
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image – install everything the training job needs
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-reproduction", image=image)

# Persistent volume to save checkpoints between runs
volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = Path("/checkpoints")

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
CFG = dict(
    embed_dim=256,
    batch_size=256,
    epochs=40,
    lr=5e-4,
    weight_decay=0.1,
    warmup_epochs=1,
    max_text_length=64,
    val_size=1000,
    log_matrix_every=100,   # steps between similarity-matrix heatmap logs
    save_every=5,            # epochs between checkpoint saves
    num_workers=4,
    seed=42,
)


# ---------------------------------------------------------------------------
# Training utilities (run inside Modal)
# ---------------------------------------------------------------------------

def similarity_matrix_figure(logits, step: int):
    """Return a wandb.Image of the N×N similarity matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb

    mat = logits.detach().float().cpu().numpy()
    n = min(32, mat.shape[0])  # cap at 32×32 for readability
    mat = mat[:n, :n]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Image–Text similarity matrix  (step {step})", fontsize=11)
    ax.set_xlabel("Text index")
    ax.set_ylabel("Image index")
    # Mark diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="lime", linewidth=1.2))
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def recall_at_k(img_emb, txt_emb, k: int = 1) -> float:
    """Image→Text Recall@k: fraction of images whose paired caption is in top-k."""
    import torch
    sims = img_emb @ txt_emb.T                          # (N, N)
    ranks = sims.argsort(dim=1, descending=True)        # (N, N)
    targets = torch.arange(len(img_emb), device=img_emb.device)
    hits = (ranks[:, :k] == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60 * 60 * 6,          # 6 hr hard limit
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
    from transformers import T5Tokenizer
    from tqdm import tqdm

    from model import CLIP, count_parameters
    from data import build_loaders

    # Reproducibility
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # ---- Weights & Biases ------------------------------------------------
    wandb.init(
        project="clip-reproduction",
        config=CFG,
        resume="allow",
    )

    # ---- Model -----------------------------------------------------------
    model = CLIP(embed_dim=CFG["embed_dim"]).to(device)
    param_info = count_parameters(model)
    print(f"Parameters: {param_info}")
    wandb.config.update(param_info)

    # ---- Data ------------------------------------------------------------
    print("Loading Flickr30k …")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_loader, val_loader = build_loaders(
        ds, tokenizer,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        val_size=CFG["val_size"],
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CFG["epochs"]
    print(f"Train batches/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    # ---- Optimiser & scheduler -------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0

    # ---- Optional resume -------------------------------------------------
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Build scheduler over the *remaining* steps so it always fits cleanly.
    # CosineAnnealingLR has no hard total_steps limit, so resuming for more
    # epochs never causes an "out of steps" error.
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

            log_dict = {
                "train/loss": loss.item(),
                "train/temperature": out["temperature"],
                "train/lr": scheduler.get_last_lr()[0],
                "step": global_step,
            }

            # Log similarity matrix heatmap periodically
            if global_step % CFG["log_matrix_every"] == 0:
                log_dict["train/similarity_matrix"] = similarity_matrix_figure(
                    out["logits"], global_step
                )

            wandb.log(log_dict)
            global_step += 1

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / steps_per_epoch

        # ---- Validation --------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_img_emb, all_txt_emb = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    out = model(images, input_ids, attention_mask)

                val_loss += out["loss"].item()
                all_img_emb.append(out["img_emb"])
                all_txt_emb.append(out["txt_emb"])

        avg_val_loss = val_loss / len(val_loader)
        img_emb_all = torch.cat(all_img_emb)
        txt_emb_all = torch.cat(all_txt_emb)
        r1 = recall_at_k(img_emb_all, txt_emb_all, k=1)
        r5 = recall_at_k(img_emb_all, txt_emb_all, k=5)

        # Full val similarity matrix for the blog
        val_mat_fig = similarity_matrix_figure(
            (img_emb_all[:32] @ txt_emb_all[:32].T) * model.temperature,
            global_step,
        )

        wandb.log({
            "val/loss": avg_val_loss,
            "val/R@1": r1,
            "val/R@5": r5,
            "val/similarity_matrix": val_mat_fig,
            "epoch": epoch + 1,
        })

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={avg_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"R@1={r1:.3f} | R@5={r5:.3f} | τ={model.temperature:.3f} | "
            f"time={epoch_time:.0f}s"
        )

        # ---- Checkpoint --------------------------------------------------
        if (epoch + 1) % CFG["save_every"] == 0 or epoch == CFG["epochs"] - 1:
            ckpt_path = CHECKPOINT_DIR / f"clip_epoch{epoch+1:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": CFG,
            }, ckpt_path)
            volume.commit()
            print(f"Saved checkpoint → {ckpt_path}")

    # Save final model weights only (for the demo)
    final_path = CHECKPOINT_DIR / "clip_final.pt"
    torch.save({"model": model.state_dict(), "config": CFG}, final_path)
    volume.commit()
    print(f"Training complete. Final model saved to {final_path}")
    wandb.finish()


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(resume_from: str = ""):
    train.remote(resume_from=resume_from)
