# Reproducing CLIP from Scratch

A from-scratch reproduction of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) trained on Flickr30k, scaling from a small T5-small + ResNet-18 model to a larger DistilBERT + ResNet-50 model. All training runs on [Modal](https://modal.com) for a few dollars.

**Blog post:** [ehersch.github.io](https://ehersch.github.io/)

---

## Architecture

| Config | Image encoder | Text encoder | Embed dim |
|--------|--------------|--------------|-----------|
| `small` | ResNet-18 (ImageNet) | T5-small encoder | 256 |
| `large` | ResNet-50 (ImageNet) | DistilBERT | 512 |
| `huge`  | ViT-S/16 (ImageNet-21k) | all-mpnet-base-v2 | 512 |

Both encoders are pretrained and fine-tuned end-to-end with symmetric InfoNCE loss over batches of 256 image-text pairs.

---

## Results (Flickr30k, 1000 val images)

| Model | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| Small | 1.9% | 10.0% | 17.2% |
| Large | 5.2% | 20.5% | 32.7% |

Random chance baseline: R@1 = 0.1%. The large model is 52× better than random.

---

## Setup

```bash
pip install -r requirements.txt

# Store W&B key in Modal
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

---

## Training

```bash
# Small model — T4, ~4 hrs, ~$3
modal run --detach train.py

# Large model — A100, ~1.5 hrs, ~$4.50
modal run --detach train_large.py

# Huge model (ViT + MPNet on CC3M) — A100, ~3 hrs, ~$9
modal run train_huge.py --prepare          # download CC3M images (once)
modal run --detach train_huge.py::pack_cc3m  # pack into tar shards (once)
modal run --detach train_huge.py           # train

# Resume from checkpoint
modal run --detach train_large.py --resume-from /checkpoints/large_epoch20.pt
```

---

## Evaluation

```bash
# Run eval + generate figures for blog
modal run eval.py --model large

# Download figures
modal volume get clip-checkpoints figures/large ./figures/large
```

---

## Queries

```bash
# Text → Image
modal run query.py --query "a dog playing in the snow"

# Image → Text
modal run query.py --image-path example_images/ex_1.jpg

# Run all example queries for both models
modal run run_examples.py
```

---

## Zero-shot classification

```bash
# CIFAR-100 and Food-101, both models in parallel
modal run zeroshot_eval.py

# Single dataset / model
modal run zeroshot_eval.py --model large --dataset food101
```

---

## Files

| File | Description |
|------|-------------|
| `model.py` | CLIP model — all three configs |
| `data.py` | Flickr30k dataset + transforms |
| `train.py` | Small model training (Modal) |
| `train_large.py` | Large model training (Modal) |
| `train_huge.py` | Huge model + CC3M data pipeline |
| `eval.py` | Recall@k + similarity matrix figures |
| `query.py` | Interactive text→image and image→text |
| `run_examples.py` | Batch queries for both models |
| `zeroshot_eval.py` | Zero-shot CIFAR-100 / Food-101 |
| `generate_tsne.py` | t-SNE embedding visualization |
| `blog/` | Blog post (HTML) |
| `example_images/` | Query images for image→text demo |
