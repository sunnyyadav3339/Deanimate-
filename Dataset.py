import torch
from PIL import Image
import os 
import config
from torch.utils.data import Dataset
import numpy as np


class AnimeHumanDataset(Dataset):
    def __init__(self, root_human, root_anime, transform=None):
        self.root_human = root_human
        self.root_anime = root_anime
        self.transform = transform

        self.human_images = os.listdir(root_human)
        self.anime_images = os.listdir(root_anime)
        self.length_dataset = max(len(self.human_images), len(self.anime_images))
        self.Human_len= len(self.human_images)
        self.Anime_len= len(self.anime_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        human_image = self.human_images[index % self.Human_len]
        anime_image = self.anime_images[index % self.Anime_len]

        human_path = os.path.join(self.root_human, human_image)
        anime_path = os.path.join(self.root_anime, anime_image)

        human_image= np.array(Image.open(human_path).convert("RGB"))
        anime_image= np.array(Image.open(anime_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=human_image, image0=anime_image)
            human_image = augmentations["image"]
            anime_image = augmentations["image0"]

        return human_image, anime_image




