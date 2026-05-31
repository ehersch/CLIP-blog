"""
Generate explainer figures for the contrastive learning section of the blog.
Run locally:  python generate_explainer_figures.py

Produces in ./assets/figures/:
  contrastive_concept.png   — pull/push intuition diagram
  embedding_space.png       — 2D shared embedding space
  similarity_matrix_explain.png — annotated N×N matrix walkthrough
  training_loop.png         — one training step diagram
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("assets/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style ────────────────────────────────────────────────────────────
BG       = "#ffffff"
SURFACE  = "#f7f7f5"
BORDER   = "#e5e5e3"
TEXT     = "#1a1a1a"
MUTED    = "#6b6b6b"
LIGHT    = "#aaaaaa"
GREEN    = "#16a34a"
RED      = "#dc2626"
BLUE     = "#2563eb"
ORANGE   = "#ea580c"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.facecolor":   BG,
    "figure.facecolor": BG,
    "text.color":       TEXT,
    "axes.edgecolor":   BORDER,
    "axes.linewidth":   0.8,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
})


# ── 1. Contrastive concept: pull & push ─────────────────────────────────────
def fig_contrastive_concept():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.08)

    titles = ["Before training", "After training"]
    # Before: scattered pairs.  After: matched pairs clustered
    np.random.seed(1)
    img_pts_before = np.random.randn(5, 2) * 1.4
    txt_pts_before = np.random.randn(5, 2) * 1.4
    # After: text points close to their image partner
    img_pts_after = np.array([[-2.0,  1.2], [ 1.5,  1.8],
                               [ 0.2, -1.5], [-1.2, -0.8], [ 2.1, -0.3]])
    txt_pts_after = img_pts_after + np.random.randn(5, 2) * 0.18

    colors = ["#2563eb","#16a34a","#ea580c","#9333ea","#0891b2"]
    labels = ["dog","beach","bike","cat","crowd"]

    for ax, title, img_pts, txt_pts in zip(
        axes, titles,
        [img_pts_before, img_pts_after],
        [txt_pts_before, txt_pts_after],
    ):
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="600",
                     color=TEXT, pad=10)

        # draw connecting lines
        for i in range(5):
            lw   = 1.2 if title == "Before training" else 2.0
            col  = LIGHT if title == "Before training" else colors[i]
            alpha= 0.4 if title == "Before training" else 0.55
            ax.plot([img_pts[i,0], txt_pts[i,0]],
                    [img_pts[i,1], txt_pts[i,1]],
                    color=col, lw=lw, alpha=alpha, zorder=1)

        # image points (circle)
        for i, (x, y) in enumerate(img_pts):
            ax.scatter(x, y, s=160, color=colors[i],
                       zorder=3, linewidths=1.5,
                       edgecolors="white")
            ax.text(x, y - 0.36, f"img: {labels[i]}",
                    ha="center", va="top", fontsize=8,
                    color=colors[i], fontweight="500")

        # text points (diamond marker)
        for i, (x, y) in enumerate(txt_pts):
            ax.scatter(x, y, s=160, color=colors[i],
                       marker="D", zorder=3, linewidths=1.5,
                       edgecolors="white", alpha=0.85)
            ax.text(x, y + 0.30, f'"{labels[i]}"',
                    ha="center", va="bottom", fontsize=8,
                    color=colors[i], style="italic")

        # legend (first panel only)
        if title == "Before training":
            ax.scatter([], [], s=100, color=MUTED, label="Image embedding")
            ax.scatter([], [], s=100, color=MUTED, marker="D", label="Text embedding")
            ax.legend(fontsize=9, loc="lower right",
                      frameon=True, framealpha=0.9,
                      edgecolor=BORDER)

        # annotation arrows for after panel
        if title == "After training":
            ax.text(0, -2.55,
                    "Matched pairs cluster together  ·  "
                    "Unmatched pairs stay apart",
                    ha="center", fontsize=9, color=MUTED, style="italic")

    # divider line between panels
    fig.add_artist(plt.Line2D(
        [0.505, 0.505], [0.08, 0.92],
        transform=fig.transFigure,
        color=BORDER, lw=1
    ))

    # overall title
    fig.suptitle("Contrastive learning: what the model is optimising",
                 fontsize=14, fontweight="600", y=1.01, color=TEXT)

    fig.savefig(OUT / "contrastive_concept.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓ contrastive_concept.png")


# ── 2. Shared embedding space (2D) ──────────────────────────────────────────
def fig_embedding_space():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Shared 256-dim embedding space (2D projection)",
                 fontsize=13, fontweight="600", color=TEXT, pad=12)

    # Subtle unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta)*2.8, np.sin(theta)*2.8,
            color=BORDER, lw=0.8, ls="--", zorder=0)
    ax.text(0, -3.1, "L2-normalised → unit hypersphere",
            ha="center", fontsize=8.5, color=LIGHT, style="italic")

    clusters = [
        dict(label="dog",    angle=20,  color="#2563eb",
             img_cap="A dog runs\nthrough grass",
             img_cap2="Two dogs playing\noutside"),
        dict(label="beach",  angle=85,  color="#16a34a",
             img_cap="Waves crash on\na sandy shore",
             img_cap2="Children playing\nat the beach"),
        dict(label="sport",  angle=155, color="#ea580c",
             img_cap="A player kicks\nthe ball",
             img_cap2="Soccer match\nin the stadium"),
        dict(label="food",   angle=215, color="#9333ea",
             img_cap="A plate of pasta\nwith sauce",
             img_cap2="Chef prepares\na meal"),
        dict(label="city",   angle=290, color="#0891b2",
             img_cap="Busy street at\nnight",
             img_cap2="People walking\nin the city"),
    ]

    r_img = 2.2
    r_txt = 1.65

    for c in clusters:
        a  = np.radians(c["angle"])
        xi = r_img * np.cos(a)
        yi = r_img * np.sin(a)
        xt = r_txt * np.cos(a + np.radians(5))
        yt = r_txt * np.sin(a + np.radians(5))
        xi2 = r_img * np.cos(a - np.radians(8))
        yi2 = r_img * np.sin(a - np.radians(8))

        # connecting line image→text
        ax.plot([xi, xt], [yi, yt], color=c["color"],
                lw=1.2, alpha=0.35, zorder=1)
        ax.plot([xi2, xt], [yi2, yt], color=c["color"],
                lw=1.2, alpha=0.35, zorder=1)

        # image dots
        ax.scatter(xi,  yi,  s=120, color=c["color"],
                   zorder=4, edgecolors="white", linewidths=1.5)
        ax.scatter(xi2, yi2, s=120, color=c["color"],
                   zorder=4, edgecolors="white", linewidths=1.5, alpha=0.75)

        # text dot (diamond)
        ax.scatter(xt, yt, s=100, color=c["color"],
                   marker="D", zorder=4,
                   edgecolors="white", linewidths=1.5)

        # caption labels
        offset = 0.32
        ax.text(xi + offset * np.cos(a),
                yi + offset * np.sin(a),
                c["img_cap"], fontsize=7.2, ha="center",
                color=c["color"], linespacing=1.4)

    # legend
    ax.scatter([], [], s=90, color=MUTED, label="Image embedding")
    ax.scatter([], [], s=90, color=MUTED, marker="D", label="Text embedding")
    ax.legend(fontsize=9, loc="center", frameon=True,
              framealpha=0.95, edgecolor=BORDER,
              bbox_to_anchor=(0.5, 0.5))

    fig.savefig(OUT / "embedding_space.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓ embedding_space.png")


# ── 3. Annotated N×N similarity matrix ──────────────────────────────────────
def fig_similarity_matrix_explain():
    n = 5
    np.random.seed(7)

    # Construct a "good" matrix: high diagonal, lower off-diagonal
    mat = np.random.uniform(0.05, 0.35, (n, n))
    np.fill_diagonal(mat, np.random.uniform(0.72, 0.92, n))

    labels = ["dog photo", "beach photo", "bike photo",
              "cat photo", "crowd photo"]
    captions = ['"a dog in\nthe snow"', '"waves on\na beach"',
                '"child on\na bicycle"', '"a cat\nsitting"',
                '"busy city\nstreet"']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_facecolor(BG)

    im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="auto")

    # Cell text
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            col = "white" if val > 0.55 else TEXT
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=col, fontweight=weight)

    # Highlight diagonal
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor=GREEN, facecolor="none", zorder=5
        )
        ax.add_patch(rect)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(captions, fontsize=8.5, color=MUTED, linespacing=1.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9, color=MUTED)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Captions (text embeddings)", fontsize=10,
                  color=MUTED, labelpad=10)
    ax.set_ylabel("Images (image embeddings)", fontsize=10,
                  color=MUTED, labelpad=10)

    # Annotations
    ax.annotate("Positive pairs:\nhigh similarity\n(should be ~1.0)",
                xy=(0, 0), xytext=(1.7, -0.7),
                fontsize=8.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
                ha="left")

    ax.annotate("Negative pairs:\nlow similarity\n(should be ~0.0)",
                xy=(2, 0), xytext=(3.5, -0.7),
                fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                ha="left")

    ax.set_title("N×N cosine similarity matrix  (N = batch size)\n"
                 "Loss pushes diagonal → 1 and off-diagonal → 0",
                 fontsize=12, fontweight="600", color=TEXT, pad=40)

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                 label="Cosine similarity")
    plt.tight_layout()
    fig.savefig(OUT / "similarity_matrix_explain.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓ similarity_matrix_explain.png")


# ── 4. One training step (pipeline diagram) ─────────────────────────────────
def fig_training_loop():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor(BG)

    def box(x, y, w, h, label, sublabel="", color=SURFACE, lw=1.0):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              linewidth=lw, edgecolor=BORDER,
                              facecolor=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=10, fontweight="600", color=TEXT, zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    fontsize=8, color=MUTED, zorder=3)

    def arrow(x1, y, x2, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=MUTED, lw=1.1),
                    zorder=4)
        if label:
            ax.text((x1+x2)/2, y + 0.18, label,
                    ha="center", fontsize=8, color=MUTED)

    # ── Row 1: image path ────────────────────────────────────────────────
    Y_IMG = 2.3
    box(0.2, Y_IMG, 1.4, 0.9, "Image", "Flickr30k")
    arrow(1.6, Y_IMG+0.45, 2.3, "224×224")
    box(2.3, Y_IMG, 1.6, 0.9, "ResNet-18\n/ ResNet-50", "", color="#f0f9ff")
    arrow(3.9, Y_IMG+0.45, 4.6, "512-d / 2048-d")
    box(4.6, Y_IMG, 1.5, 0.9, "Linear\nprojection", "→ 256-d / 512-d")
    arrow(6.1, Y_IMG+0.45, 6.8, "L2 norm")
    box(6.8, Y_IMG, 1.4, 0.9, "Image\nembedding", "I_e  (B×D)")

    # ── Row 2: text path ─────────────────────────────────────────────────
    Y_TXT = 0.8
    box(0.2, Y_TXT, 1.4, 0.9, "Caption", "Flickr30k")
    arrow(1.6, Y_TXT+0.45, 2.3, "tokens")
    box(2.3, Y_TXT, 1.6, 0.9, "T5-small\n/ DistilBERT", "", color="#fdf4ff")
    arrow(3.9, Y_TXT+0.45, 4.6, "mean-pool")
    box(4.6, Y_TXT, 1.5, 0.9, "Linear\nprojection", "→ 256-d / 512-d")
    arrow(6.1, Y_TXT+0.45, 6.8, "L2 norm")
    box(6.8, Y_TXT, 1.4, 0.9, "Text\nembedding", "T_e  (B×D)")

    # ── Merge into loss ──────────────────────────────────────────────────
    mid_y = (Y_IMG + Y_TXT) / 2 + 0.45
    ax.plot([8.2, 8.7], [Y_IMG+0.45, mid_y], color=MUTED, lw=1.1)
    ax.plot([8.2, 8.7], [Y_TXT+0.45, mid_y], color=MUTED, lw=1.1)
    ax.annotate("", xy=(8.7, mid_y), xytext=(8.2, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))

    box(8.7, mid_y-0.45, 1.6, 0.9, "Contrastive\nloss", "symmetric InfoNCE",
        color="#fff7ed", lw=1.2)

    arrow(10.3, mid_y, 10.75, "")
    ax.text(10.8, mid_y, "∇", fontsize=20, va="center",
            color=MUTED, fontweight="300")

    # ── Batch label ──────────────────────────────────────────────────────
    ax.text(0.2, 3.65,
            "One training step  ·  batch size = 256  ·  "
            "each image paired with one randomly sampled caption",
            fontsize=8.5, color=MUTED, style="italic")

    # ── Temperature label ────────────────────────────────────────────────
    ax.text(8.0, 3.65, "× τ  (learnable temperature)",
            fontsize=8.5, color=MUTED, style="italic")

    fig.savefig(OUT / "training_loop.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓ training_loop.png")


if __name__ == "__main__":
    fig_contrastive_concept()
    fig_embedding_space()
    fig_similarity_matrix_explain()
    fig_training_loop()
    print(f"\nAll figures saved to {OUT.resolve()}")
