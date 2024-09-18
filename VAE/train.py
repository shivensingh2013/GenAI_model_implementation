## Train a Vanilla VAE using MNIST fashion dataset

import torch
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from  dataloader import custom_dataset


if __name__ == "__main__":

    ## Load dataset 
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    dataset_obj = custom_dataset(folder)
    train_set, val_set = torch.utils.data.random_split(dataset_obj,[50000, 10000])
    train_dataload = DataLoader(dataset =train_set,batch_size = 4,shuffle = True ) 
    val_dataload = DataLoader(dataset =val_set,batch_size = 4,shuffle = True )
    print(train_dataload.__len__())

    ## Training code
    







    



