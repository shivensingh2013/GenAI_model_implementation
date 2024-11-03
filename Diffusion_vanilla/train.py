## Train a Diffusion model using MNIST fashion dataset

import torch
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from  dataloader import custom_dataset
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim

'''
Training Process for DDPM:

1)Take a randomly sampled data point from our training dataset
2)Select a random timestep on our noise (variance) schedule
3)Add the noise from that time step to our data, simulating the forward diffusion process through the “diffusion kernel”
4)Pass our defused image into our model to predict the noise we added
5) Compute the mean squared error between the predicted noise and the actual noise and optimize our model’s parameters through that objective function
6)And repeat!

'''

class sinusoidal_embeddings(nn.module):
    def __init__(self, time_step:int,embed_dim:int):
        super().__init__()
        position = torch.arange(time_step).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings
    
    def forward(self,x,t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]





def train_one_epoch():
    return null



if __name__ == "__main__":
    ## loading the dataloader
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    diffusion_data = custom_dataset(folder)
    print(diffusion_data.__len__)

    ## creating dataloder from Dataset
    train_data, val_data = torch.utils.data.random_split(diffusion_data,[50000,10000])
    train_loader = Dataloader(dataset =train_data,batch_size = 1 ,shuffle = True )
    val_loader = Dataloader(dataset =val_data,batch_size = 1 ,shuffle = True )
    
    ## Model Architecture


