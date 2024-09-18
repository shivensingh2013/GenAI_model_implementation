import torch.nn as nn
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataloader import custom_dataset
import numpy as np
import torch
class vae_arch(nn.Module):
    ## Input - 28 *28 image with values 0-255 

    def __init__(self):
        super().__init__()
        self.z_layer = 16

        self.encoder_layer = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,self.z_layer))
        
        self.decoder_layer = nn.Sequential(
                 nn.Linear(self.z_layer,256),
                    nn.ReLU(),
                    nn.Linear(256,784),
                    nn.Sigmoid()
                
        )
    def forward(self,inp_img):
        ##transform the input image into feedable input
        
        z = self.encoder_layer(inp_img)
        recon_img = self.decoder_layer(z)
        return recon_img



if __name__ == "__main__" : 
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    dataset_obj = custom_dataset(folder)
    sample_dataloader = DataLoader(dataset_obj,batch_size =4)
    sample = next(iter(sample_dataloader))
    label,image = sample[0,1],sample[0,1:]
    image = image.type(torch.uint8)
    print(image.shape,image.dtype)
    # vae_obj = vae_arch()
    # target_img = vae_obj.forward(image)
    # print(target_img)

    



