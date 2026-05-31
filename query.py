"""
Two query directions — both run on Modal.

TEXT → IMAGE  (give text, get matching images)
    modal run query.py --query "a dog playing in the snow"
    modal run query.py --query "sunset over the ocean" --top-k 8

IMAGE → TEXT  (give an image path, get matching captions)
    modal run query.py --image-path /path/to/your/photo.jpg
    modal run query.py --image-path /path/to/photo.jpg --top-k 10

Results are saved as PNGs in the current directory.
Run `modal run eval.py` first to build the image index.
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("model", "data")
)

app = modal.App("clip-query", image=image)

volume = modal.Volume.from_name("clip-checkpoints", create_if_missing=False)
CHECKPOINT_DIR = Path("/checkpoints")


# ---------------------------------------------------------------------------
# Shared: load model + tokenizer
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "small": dict(checkpoint="clip_final.pt",  clip_config="small", tokenizer="t5-small"),
    "large": dict(checkpoint="large_final.pt", clip_config="large", tokenizer="distilbert"),
    "huge":  dict(checkpoint="huge_final.pt",  clip_config="huge",  tokenizer="minilm"),
}

def _load_model_and_tokenizer(device, model_name: str = "large"):
    import torch
    from model import CLIP

    cfg = MODEL_CONFIGS[model_name]
    ckpt = torch.load(CHECKPOINT_DIR / cfg["checkpoint"], map_location=device)
    model = CLIP(config=cfg["clip_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if cfg["tokenizer"] == "t5-small":
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    elif cfg["tokenizer"] == "minilm":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tokenizer


def _load_image_index():
    import torch
    index_path = CHECKPOINT_DIR / "image_index.pt"
    if not index_path.exists():
        raise RuntimeError("Image index not found — run `modal run eval.py` first.")
    data = torch.load(index_path)
    return data["embeddings"], data["captions"]


def _load_caption_index(model, tokenizer, device, batch_size=512):
    """
    Encode every caption in Flickr30k into a (N, 256) tensor.
    Saved to volume so it only runs once.
    """
    import torch
    from datasets import load_dataset

    index_path = CHECKPOINT_DIR / "caption_index.pt"
    if index_path.exists():
        data = torch.load(index_path)
        return data["embeddings"], data["captions"], data["image_indices"]

    print("Building caption index (one-time, ~2 min)...")
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    all_emb, all_captions, all_img_indices = [], [], []

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        ids_list, masks = [], []
        for i, row in enumerate(batch):
            for cap in row["caption"]:   # all 5 captions per image
                tok = tokenizer(cap, max_length=64, padding="max_length",
                                truncation=True, return_tensors="pt")
                ids_list.append(tok["input_ids"].squeeze(0))
                masks.append(tok["attention_mask"].squeeze(0))
                all_captions.append(cap)
                all_img_indices.append(start + i)  # which dataset image this came from

        with torch.no_grad():
            emb = model.encode_text(
                torch.stack(ids_list).to(device),
                torch.stack(masks).to(device),
            )
        all_emb.append(emb.cpu())
        if start % 5000 == 0:
            print(f"  {start}/{len(ds)}")

    emb_tensor = torch.cat(all_emb)
    torch.save({
        "embeddings": emb_tensor,
        "captions": all_captions,
        "image_indices": all_img_indices,
    }, index_path)
    volume.commit()
    print(f"Caption index saved: {emb_tensor.shape}")
    return emb_tensor, all_captions, all_img_indices


# ---------------------------------------------------------------------------
# TEXT → IMAGE
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60 * 10,
    volumes={CHECKPOINT_DIR: volume},
)
def text_to_image(query: str, top_k: int = 6, model_name: str = "large") -> bytes:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from datasets import load_dataset

    device = torch.device("cuda")
    model, tokenizer = _load_model_and_tokenizer(device, model_name)
    img_emb, captions = _load_image_index()
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    # Encode query text
    tokens = tokenizer(query, max_length=64, padding="max_length",
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        txt_emb = model.encode_text(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )

    sims = (img_emb.to(device) @ txt_emb.T).squeeze(1).cpu()
    top_idx = sims.argsort(descending=True)[:top_k].tolist()
    top_scores = sims[top_idx].tolist()

    fig, axes = plt.subplots(1, top_k, figsize=(3.2 * top_k, 3.8))
    fig.suptitle(f'Text → Image  |  "{query}"',
                 fontsize=13, fontweight="bold", y=1.03)

    for rank, (ax, idx, score) in enumerate(zip(axes, top_idx, top_scores)):
        ax.imshow(ds[idx]["image"].convert("RGB"))
        ax.set_title(f"#{rank+1}  {score:.3f}", fontsize=9,
                     color="#27ae60" if rank == 0 else "#555")
        cap = captions[idx]
        ax.set_xlabel(cap[:55] + "…" if len(cap) > 55 else cap, fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        color = "#2ecc71" if rank == 0 else "#bbb"
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(3 if rank == 0 else 1)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# IMAGE → TEXT
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60 * 10,
    volumes={CHECKPOINT_DIR: volume},
)
def image_to_text(image_bytes: bytes, top_k: int = 6, model_name: str = "large") -> bytes:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    from datasets import load_dataset
    from data import VAL_TRANSFORM

    device = torch.device("cuda")
    model, tokenizer = _load_model_and_tokenizer(device, model_name)
    cap_emb, cap_texts, cap_img_indices = _load_caption_index(model, tokenizer, device)
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)["test"]

    # Encode the query image
    query_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = VAL_TRANSFORM(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)  # (1, 256)

    sims = (cap_emb.to(device) @ img_emb.T).squeeze(1).cpu()
    top_idx = sims.argsort(descending=True)[:top_k].tolist()
    top_scores = sims[top_idx].tolist()

    # ---- Figure layout -------------------------------------------------------
    # Left column  : query image (tall)
    # Right columns: one column per result, each split into image (top) + caption (bottom)
    #   height_ratios — image row is 3×, caption row is 1× so text has real estate
    import textwrap

    n_cols = 1 + top_k                        # query col + result cols
    fig = plt.figure(figsize=(3.5 + 3.2 * top_k, 8))
    fig.suptitle("Image → Text", fontsize=14, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(
        2, n_cols,
        height_ratios=[3, 1.4],   # image row taller; caption row has breathing room
        hspace=0.08,
        wspace=0.12,
    )

    # Query image — spans both rows
    ax_query = fig.add_subplot(gs[:, 0])
    ax_query.imshow(query_img)
    ax_query.set_title("Query\nimage", fontsize=10, fontweight="bold", pad=6)
    ax_query.set_xticks([]); ax_query.set_yticks([])
    for s in ax_query.spines.values():
        s.set_edgecolor("#3498db"); s.set_linewidth(2.5)

    for rank, (cap_idx, score) in enumerate(zip(top_idx, top_scores)):
        col = rank + 1
        caption_text = cap_texts[cap_idx]
        source_img_idx = cap_img_indices[cap_idx]
        color = "#2ecc71" if rank == 0 else "#aaa"

        # ---- Image cell ----
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(ds[source_img_idx]["image"].convert("RGB"))
        ax_img.set_title(f"#{rank+1}   sim = {score:.3f}", fontsize=9,
                         color="#27ae60" if rank == 0 else "#666", pad=5)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for s in ax_img.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2.5 if rank == 0 else 1)

        # ---- Caption cell ----
        ax_cap = fig.add_subplot(gs[1, col])
        ax_cap.axis("off")
        # Hard-wrap at 32 chars so each line fits the column width
        wrapped = "\n".join(textwrap.wrap(caption_text, width=32))
        ax_cap.text(0.5, 0.95, wrapped, fontsize=8, va="top", ha="center",
                    transform=ax_cap.transAxes, linespacing=1.5,
                    color="#111", style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                              edgecolor=color, linewidth=1.2))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    query: str = "",
    image_path: str = "",
    top_k: int = 6,
):
    if query and image_path:
        print("Pass either --query or --image-path, not both.")
        return

    if query:
        print(f"Text → Image  |  '{query}'  (top {top_k})")
        png_bytes = text_to_image.remote(query=query, top_k=top_k)
        slug = query.lower().replace(" ", "_")[:40]
        out = Path(f"result_t2i_{slug}.png")

    elif image_path:
        p = Path(image_path)
        if not p.exists():
            print(f"File not found: {image_path}")
            return
        print(f"Image → Text  |  {p.name}  (top {top_k})")
        png_bytes = image_to_text.remote(
            image_bytes=p.read_bytes(), top_k=top_k
        )
        out = Path(f"result_i2t_{p.stem}.png")

    else:
        print("Usage:\n"
              "  modal run query.py --query 'a dog in snow'\n"
              "  modal run query.py --image-path /path/to/photo.jpg")
        return

    out.write_bytes(png_bytes)
    print(f"Saved → {out.resolve()}")
