## Train a Vanilla VAE using MNIST fashion dataset

import torch
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from  dataloader import custom_dataset
from model import vae_arch
from torch.optim import Adam
import torch.nn as nn

if __name__ == "__main__":

    ## Load dataset 
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    dataset_obj = custom_dataset(folder)
    train_set, val_set = torch.utils.data.random_split(dataset_obj,[50000, 10000])
    train_dataload = DataLoader(dataset =train_set,batch_size = 8,shuffle = True ) 
    val_dataload = DataLoader(dataset =val_set,batch_size = 8,shuffle = True )
    # print(train_dataload.__len__())

    ## Loading the Model architecture    
    model = vae_arch()
    learning_rate = 0.001   
    num_epoch = 100
    ##optimizer
    optimizer = Adam(model.parameters(),lr = learning_rate)

    ## Loss function
    criterion = nn.MSELoss()

    ## start the training process:
    model.train()
    
    for epoch in range(num_epoch):
        total_loss = 0.0
        for label_image in train_dataload:
            image = label_image[:,1:]
            ## transformation on the image
            image = image/255
            optimizer.zero_grad()
            loss = criterion(image.float() , recon.float())
            total_loss +=np.float(loss.item())
            loss.backward()
            optimizer.step()
            
        # Print the epoch loss
        epoch_loss = total_loss / len(train_dataload.dataset)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epoch, epoch_loss)
        )







        










    



