import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from scipy import spatial
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import timm
from timm.utils import AverageMeter
import sys
sys.path.append('/mnt/sdd/syz/kaggle/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer
import warnings




class DiffusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath'])
        image = self.transform(image)
        prompt = row['prompt']
        return image, prompt


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            '/mnt/sdd/syz/kaggle/sentence-transformers-222/all-MiniLM-L6-v2',
            device='cpu'
        )
    
    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, 
            show_progress_bar=False, 
            convert_to_tensor=True
        )
        return images, prompt_embeddings
    


def get_dataloaders(
    trn_df,
    val_df,
    input_size,
    batch_size
):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trn_dataset = DiffusionDataset(trn_df, transform)
    val_dataset = DiffusionDataset(val_df, transform)
    collator = DiffusionCollator()
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collator
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
        collate_fn=collator
    )
    return dataloaders



