from torch.utils.data import Dataset, Sampler, DataLoader
import pytorch_lightning as pl
from clear_pipe.sd import StableDiffusion
from typing import Optional, List
import os
from collections import defaultdict
import tqdm
from PIL import Image
import torch
import numpy as np
from dataclasses import dataclass, field
import random


@dataclass
class DatasetEntry:
    filetext: str
    placeholder: str
    text_templates: List[str] = field(default_factory=lambda: ["[filewords]"])
    npimage: Optional[np.ndarray] = None
    latent: Optional[torch.Tensor] = None
    shuffle_tags: bool = False

    def get_latent(self, vae):
        if self.latent is not None:
            return self.latent
        torchdata = torch.from_numpy(self.npimage).permute(2, 0, 1)
        self.latent = (
            vae.encode(torchdata).latent_dist.sample().detach()
            * vae.config.scaling_factor
        )
        return self.latent

    def get_text(self):
        text = random.choice(self.lines)
        tags = self.filetext.split(",")
        if self.tag_drop_out != 0:
            tags = [t for t in tags if random.random() > self.tag_drop_out]
        if self.shuffle_tags:
            random.shuffle(tags)
        text = text.replace("[filewords]", ",".join(tags))
        text = text.replace("[name]", self.placeholder)
        return text


class ClearDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str,
        placeholder: Optional[str] = None,
        text_templates: Optional[List[str]] = None,
        shuffle_tags: bool = False,
    ):
        self.placeholder = placeholder or ""
        self.dataset = []

        assert data_root, "dataset directory not specified"
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        self.image_paths = [
            os.path.join(data_root, file_path) for file_path in os.listdir(data_root)
        ]

        self.shuffle_tags = shuffle_tags
        groups = defaultdict(list)

        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            try:
                image = Image.open(path)
            except Exception:
                continue

            text_filename = os.path.splitext(path)[0] + ".txt"

            try:
                with open(text_filename, "r", encoding="utf8") as file:
                    filename_text = file.read()
            except:
                print(f"Skipped image {path} (can't find caption txt)")
                continue

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            entry = DatasetEntry(
                filetext=filename_text,
                placeholder=placeholder,
                text_templates=text_templates,
                npimage=npimage,
            )

            groups[image.size].append(len(self.dataset))
            self.dataset.append(entry)

        self.groups = list(groups.values())
        assert self.dataset, "No images have been found in the dataset."

        if len(groups) > 1:
            print("Buckets:")
            for (w, h), ids in sorted(groups.items(), key=lambda x: x[0]):
                print(f"  {w}x{h}: {len(ids)}")
            print()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class GroupedBatchSampler(Sampler):
    def __init__(self, data_source: ClearDataset, batch_size: int):
        super().__init__(data_source)

        n = len(data_source)
        self.groups = data_source.groups
        self.len = n_batch = n // batch_size
        expected = [len(g) / n * n_batch * batch_size for g in data_source.groups]
        self.base = [int(e) // batch_size for e in expected]
        self.n_rand_batches = nrb = n_batch - sum(self.base)
        self.probs = [
            e % batch_size / nrb / batch_size if nrb > 0 else 0 for e in expected
        ]
        self.batch_size = batch_size

    def __len__(self):
        return self.len

    def __iter__(self):
        b = self.batch_size

        for g in self.groups:
            random.shuffle(g)

        batches = []
        for g in self.groups:
            batches.extend(g[i * b : (i + 1) * b] for i in range(len(g) // b))
        for _ in range(self.n_rand_batches):
            rand_group = random.choices(self.groups, self.probs)[0]
            batches.append(random.choices(rand_group, k=b))

        random.shuffle(batches)

        yield from batches


class BatchLoader:
    def __init__(self, data: List[DatasetEntry], vae):
        self.cond_text = [entry.get_text() for entry in data]
        self.latent = torch.stack([entry.get_latent(vae) for entry in data]).squeeze(1)

    def pin_memory(self):
        self.latent = self.latent.pin_memory()
        return self


class ClearDataModule(pl.LightningDataModule):
    def __init__(
        self, *, data_root: str, placeholder: str, batch_size: int, sd: StableDiffusion
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.dataset = ClearDataset(data_root=self.data_root, placeholder=placeholder)
        self.batch_size = batch_size
        self.vae = sd.vae

    def train_dataloader(self):
        sampler = GroupedBatchSampler(self.dataset, self.batch_size)
        return DataLoader(
            self.dataset, sampler=sampler, collate_fn=lambda d: BatchLoader(d, self.vae)
        )
