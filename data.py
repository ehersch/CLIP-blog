"""
Flickr30k dataset wrapper for CLIP training.
Each item returns one (image, caption) pair — one caption is sampled
randomly from the five available per image, matching the paper's approach.

Works with any HuggingFace tokenizer (T5Tokenizer, DistilBertTokenizer, etc.)
"""

import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Standard ImageNet normalisation used by the pretrained ResNet-18
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])


class Flickr30kDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,        # any HuggingFace tokenizer
        max_length: int = 64,
        split: str = "train",
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = TRAIN_TRANSFORM if split == "train" else VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]

        # Image: HuggingFace flickr30k stores PIL images directly
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        # Caption: sample one of the five randomly during training
        captions = row["caption"]
        caption = random.choice(captions) if isinstance(captions, list) else captions

        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
        }


def build_loaders(
    ds,
    tokenizer,  # any HuggingFace tokenizer
    batch_size: int = 256,
    num_workers: int = 4,
    val_size: int = 1000,
):
    """Split Flickr30k into train/val and return DataLoaders."""
    full = ds["test"]  # nlphuji/flickr30k only has a 'test' split (31,014 items)
    n = len(full)
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = Flickr30kDataset(full.select(train_idx), tokenizer, split="train")
    val_ds = Flickr30kDataset(full.select(val_idx), tokenizer, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
