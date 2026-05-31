"""
CLIP huge training on Modal — ViT-S/16 + all-mpnet-base-v2 → 512-dim, trained on CC3M.

Architecture:
  - Image encoder: ViT-S/16 pretrained on ImageNet-21k (~22M params)
                   Uses the [CLS] token from the final layer → linear projection → 512-dim.
                   ViT splits images into 16×16 patches and applies full self-attention over
                   all patches, capturing global scene context better than conv-based ResNets.

  - Text encoder:  all-mpnet-base-v2 (~110M params), bidirectional transformer
                   fine-tuned specifically for semantic similarity via sentence-transformers.
                   Mean-pooled over all token embeddings → linear projection → 512-dim.

                   Why not the original CLIP text encoder (GPT-style, causal)?
                   The original uses a decoder-style transformer with causal (left-to-right)
                   self-attention, taking the [EOS] token as the sentence representation.
                   This works when training from scratch on 400M pairs because the model
                   learns to compress full sentence meaning into that last token.
                   With pretrained weights, [EOS] was trained to predict the next token —
                   not summarize the sentence — so it's a poor embedding.
                   Bidirectional encoders (BERT-style) with mean-pooling give a genuinely
                   global representation: every token attends to every other in both directions.

                   Why not train from scratch?
                   The original CLIP can train from scratch because 400M pairs provide enough
                   signal to learn language grounded in visual semantics. At our scale (~800K
                   pairs from CC3M), a from-scratch encoder produces near-random embeddings.
                   Using pretrained encoders means the model starts with deep language
                   understanding and only needs to learn alignment — a much easier problem.

  - Shared embedding dim: 512 (MPNet hidden dim is 768 → projects down, which is cleaner
                   than MiniLM's 384 → projecting up)

Dataset:
  CC3M (Conceptual Captions 3M) — ~800K pairs downloaded before volume ran out of space.
  Still 26× more data than Flickr30k. Images stored as WebDataset .tar shards on the volume
  after a one-time packing step (pack_cc3m) for fast sequential streaming.

  Why CC3M vs Flickr30k:
  Data scale is the single highest-leverage change for retrieval quality. Model capacity
  was never the bottleneck — the embedding space was simply too sparse to generalize.

Usage:
  Step 1 — download CC3M images (run once):
    modal run train_huge.py --prepare

  Step 2 — pack into tar shards for fast streaming (run once, ~20 min):
    modal run --detach train_huge.py::pack_cc3m

  Step 3 — train (detached, ~3 hrs on A100, ~$9):
    modal run --detach train_huge.py

  Resume:
    modal run --detach train_huge.py --resume-from /checkpoints/huge_epoch20.pt
"""

import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("img2dataset")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-huge", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = Path("/checkpoints")

CC3M_DIR = Path("/checkpoints/cc3m")  # where img2dataset writes shards
CC3M_SHARDS = str(CC3M_DIR / "{00000..02840}.tar")  # ~2841 shards for CC3M train

CFG = dict(
    model_config="huge",
    batch_size=256,
    epochs=40,
    lr=2e-4,
    weight_decay=0.1,
    warmup_epochs=2,
    max_text_length=64,
    val_size=5000,  # larger val set given bigger dataset
    log_matrix_every=500,
    save_every=2,
    num_workers=8,
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


# ---------------------------------------------------------------------------
# CC3M WebDataset loader
# ---------------------------------------------------------------------------


def build_cc3m_loaders(tokenizer, batch_size: int, num_workers: int, val_size: int):
    """
    Streams CC3M from WebDataset .tar shards (packed by pack_cc3m).
    Returns (train_loader, val_loader, train_size_estimate).
    """
    import random
    import webdataset as wds
    from torchvision import transforms

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    VAL_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    def tokenize(caption):
        tokens = tokenizer(
            caption,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)

    def make_train(sample):
        img = TRAIN_TRANSFORM(sample["jpg"].convert("RGB"))
        ids, mask = tokenize(sample["txt"])
        return {"image": img, "input_ids": ids, "attention_mask": mask}

    def make_val(sample):
        img = VAL_TRANSFORM(sample["jpg"].convert("RGB"))
        ids, mask = tokenize(sample["txt"])
        return {"image": img, "input_ids": ids, "attention_mask": mask}

    TAR_DIR = CC3M_DIR / "tars"
    all_shards = sorted(TAR_DIR.glob("*.tar"))
    print(f"Found {len(all_shards)} tar shards")
    random.seed(42)
    random.shuffle(all_shards)

    n_val = max(1, int(len(all_shards) * 0.02))
    val_shards = [str(s) for s in all_shards[:n_val]]
    train_shards = [str(s) for s in all_shards[n_val:]]

    train_ds = (
        wds.WebDataset(train_shards, resampled=True, shardshuffle=True)
        .shuffle(2000)
        .decode("pil")
        .map(make_train)
        .batched(batch_size, partial=False)
    )
    val_ds = (
        wds.WebDataset(val_shards)
        .decode("pil")
        .map(make_val)
        .batched(batch_size, partial=True)
    )

    train_loader = wds.WebLoader(
        train_ds, batch_size=None, num_workers=num_workers, pin_memory=True
    )
    val_loader = wds.WebLoader(val_ds, batch_size=None, num_workers=2, pin_memory=True)

    # Estimate: ~1000 samples/shard × n_train_shards
    train_size = len(train_shards) * 1000
    return train_loader, val_loader, train_size


# ---------------------------------------------------------------------------
# Data preparation — run once to download CC3M images to the volume
# ---------------------------------------------------------------------------


@app.function(
    cpu=16,
    memory=32768,
    timeout=60 * 60 * 3,  # 3 hrs
    volumes={CHECKPOINT_DIR: volume},
)
def prepare_cc3m():
    """
    Downloads CC3M train images using img2dataset and saves WebDataset shards
    to /checkpoints/cc3m/. Run this ONCE before training.
    """
    import subprocess
    from datasets import load_dataset

    CC3M_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(CC3M_DIR.glob("*.tar"))
    if len(existing) > 100:
        print(f"CC3M already downloaded: {len(existing)} shards found.")
        return

    # Write a TSV of (url, caption) for img2dataset
    print("Fetching CC3M metadata from HuggingFace...")
    ds = load_dataset(
        "google-research-datasets/conceptual_captions",
        split="train",
        trust_remote_code=True,
    )
    tsv_path = "/tmp/cc3m_urls.tsv"
    with open(tsv_path, "w") as f:
        f.write("url\tcaption\n")
        for row in ds:
            url = row.get("image_url", row.get("url", ""))
            cap = row.get("caption", "")
            if url and cap:
                f.write(f"{url}\t{cap}\n")
    print(f"Wrote {len(ds)} URLs to {tsv_path}")

    # Download with img2dataset
    print("Downloading images (this takes ~45 min)...")
    cmd = [
        "img2dataset",
        "--url_list",
        tsv_path,
        "--input_format",
        "tsv",
        "--url_col",
        "url",
        "--caption_col",
        "caption",
        "--output_type",
        "webdataset",
        "--output_folder",
        str(CC3M_DIR),
        "--image_size",
        "256",
        "--resize_mode",
        "keep_ratio",
        "--min_image_size",
        "64",
        "--max_aspect_ratio",
        "3.0",
        "--number_sample_per_shard",
        "1000",
        "--processes_count",
        "16",
        "--thread_count",
        "64",
        "--retries",
        "2",
        "--timeout",
        "10",
        "--enable_wandb",
        "False",
    ]
    subprocess.run(cmd, check=True)
    volume.commit()

    shards = list(CC3M_DIR.glob("*.tar"))
    print(f"Done. {len(shards)} shards saved to {CC3M_DIR}")


@app.function(
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 2,
    volumes={CHECKPOINT_DIR: volume},
    max_containers=16,
)
def pack_subdirs(subdir_names: list[str], worker_id: int) -> int:
    """Pack a chunk of subdirectories into tar shards. Run in parallel via pack_cc3m."""
    import tarfile
    import io

    TAR_DIR = CC3M_DIR / "tars"
    TAR_DIR.mkdir(parents=True, exist_ok=True)

    SHARD_SIZE = 1000
    buffer = []
    shard_idx = worker_id * 50  # offset so workers don't collide on shard filenames
    shards_written = 0

    def write_shard(buf, idx):
        tar_path = TAR_DIR / f"{idx:05d}.tar"
        with tarfile.open(tar_path, "w") as tf:
            for stem, jpg_bytes, txt in buf:
                jpg_info = tarfile.TarInfo(name=f"{stem}.jpg")
                jpg_info.size = len(jpg_bytes)
                tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
                txt_bytes = txt.encode()
                txt_info = tarfile.TarInfo(name=f"{stem}.txt")
                txt_info.size = len(txt_bytes)
                tf.addfile(txt_info, io.BytesIO(txt_bytes))

    for subdir_name in subdir_names:
        subdir = CC3M_DIR / subdir_name
        if not subdir.exists():
            continue
        subdir_files = []
        for jpg_path in sorted(subdir.glob("*.jpg")):
            txt_path = jpg_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            try:
                jpg_bytes = jpg_path.read_bytes()
                txt_str = txt_path.read_text().strip()
                if not txt_str:
                    continue
                stem = f"{subdir_name}_{jpg_path.stem}"
                buffer.append((stem, jpg_bytes, txt_str))
                subdir_files.extend([jpg_path, txt_path])
                json_path = jpg_path.with_suffix(".json")
                if json_path.exists():
                    subdir_files.append(json_path)
            except Exception:
                continue

            if len(buffer) >= SHARD_SIZE:
                write_shard(buffer, shard_idx)
                shard_idx += 1
                shards_written += 1
                buffer = []

        for f in subdir_files:
            try:
                f.unlink()
            except Exception:
                pass
        try:
            subdir.rmdir()
        except Exception:
            pass

    if buffer:
        write_shard(buffer, shard_idx)
        shards_written += 1

    volume.commit()
    print(
        f"Worker {worker_id}: packed {shards_written} shards from {len(subdir_names)} subdirs"
    )
    return shards_written


@app.function(
    cpu=2,
    memory=4096,
    timeout=60 * 60 * 3,
    volumes={CHECKPOINT_DIR: volume},
)
def pack_cc3m():
    """
    Coordinates parallel packing of cc3m subdirs into WebDataset tar shards.
    Splits subdirs across 16 workers — completes in ~10 min instead of 36 hrs.
    """
    import math

    subdirs = sorted(
        [d.name for d in CC3M_DIR.iterdir() if d.is_dir() and d.name != "tars"]
    )
    existing_tars = len(list((CC3M_DIR / "tars").glob("*.tar")))
    print(f"Already have {existing_tars} tars. Packing remaining {len(subdirs)} subdirs...")

    n_workers = min(16, len(subdirs))
    chunk_size = math.ceil(len(subdirs) / n_workers)
    chunks = [subdirs[i : i + chunk_size] for i in range(0, len(subdirs), chunk_size)]

    total = sum(pack_subdirs.starmap([(chunk, i) for i, chunk in enumerate(chunks)]))

    volume.commit()
    tar_count = len(list((CC3M_DIR / "tars").glob("*.tar")))
    print(f"Done. {total} shards written, {tar_count} tars in {CC3M_DIR / 'tars'}")


def recall_at_k(img_emb, txt_emb, k: int = 1) -> float:
    import torch

    sims = img_emb @ txt_emb.T
    ranks = sims.argsort(dim=1, descending=True)
    targets = torch.arange(len(img_emb), device=img_emb.device)
    hits = (ranks[:, :k] == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


@app.function(
    gpu="A100",
    timeout=60 * 60 * 8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(resume_from: str = ""):
    import random
    import numpy as np
    import torch
    import wandb
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import AutoTokenizer
    from tqdm import tqdm

    from model import CLIP, count_parameters

    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    wandb.init(
        project="clip-reproduction",
        config={**CFG, "model": "huge"},
        tags=["huge", "vit-s16", "mpnet"],
        resume="allow",
    )

    # ---- Model -----------------------------------------------------------
    model = CLIP(config="huge").to(device)
    params = count_parameters(model)
    print(f"Parameters: {params}")
    wandb.config.update(params)

    # ---- Data ------------------------------------------------------------
    TAR_DIR = CC3M_DIR / "tars"
    if not TAR_DIR.exists() or len(list(TAR_DIR.glob("*.tar"))) < 10:
        raise RuntimeError(
            "CC3M tars not found. Run packing step first:\n"
            "  modal run --detach train_huge.py::pack_cc3m"
        )

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    train_loader, val_loader, train_size = build_cc3m_loaders(
        tokenizer,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        val_size=CFG["val_size"],
    )

    steps_per_epoch = len(train_loader)
    print(f"CC3M train pairs: {train_size:,}  |  Steps/epoch: {steps_per_epoch}")

    # ---- Optimizer -------------------------------------------------------
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
            path = CHECKPOINT_DIR / f"huge_epoch{epoch+1:02d}.pt"
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

    final = CHECKPOINT_DIR / "huge_final.pt"
    torch.save({"model": model.state_dict(), "config": CFG}, final)
    volume.commit()
    print(f"Done. Final model → {final}")
    wandb.finish()


@app.local_entrypoint()
def main(resume_from: str = "", prepare: bool = False):
    if prepare:
        print("Spawning CC3M download — safe to close your terminal.")
        print("Monitor with: modal app logs clip-huge")
        call = prepare_cc3m.spawn()
        print(f"Job ID: {call.object_id}")
    else:
        train.remote(resume_from=resume_from)


# To pack after downloading:
#   modal run --detach train_huge.py::pack_cc3m
# To train after packing:
#   modal run --detach train_huge.py
