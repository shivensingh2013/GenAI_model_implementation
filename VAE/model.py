import torch.nn as nn
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataloader import custom_dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

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
        self.mu_layer = nn.Linear(self.z_layer, self.z_layer)
        self.log_var_layer = nn.Linear(self.z_layer, self.z_layer)
    

    def forward(self,inp_img):
        ## From encoder , we get the mu and sigma vectors depending on the size of Z vector
        z = self.encoder_layer(inp_img)
        mu = self.mu_layer(z)
        log_var = self.log_var_layer(z)
        std = torch.exp(0.5 * log_var)
        ## random value vector from gaussian distribution
        sample_z = torch.randn_like(std)
        
        ## Obtain a Zi  = mu_i + sigma_i * (Normal distribution(0,1)) 
        encoded = mu + std * sample_z
        recon_img = self.decoder_layer(encoded)
        return encoded,recon_img

    def sample(self):
        z = torch.randn(1,self.z_layer)
        recon = self.decoder_layer(z)
        return recon


if __name__ == "__main__" : 
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    dataset_obj = custom_dataset(folder)
    sample_dataloader = DataLoader(dataset_obj,batch_size =4)
    sample = next(iter(sample_dataloader))
    label,image = sample[:,0],sample[:,1:]
    ## transformation
    image = image/255
    vae_obj = vae_arch()
    z,target_img = vae_obj.forward(image)

    ## sampling from the dataset
    gen_image = vae_obj.sample()
    gen_image = torch.round(gen_image *255)
    gen_image = gen_image.detach().numpy().astype(np.uint8)
    gen_image = gen_image.reshape((28,28))
    buf = Image.fromarray(gen_image)
    buf.save("generated_sample.png")
    print(gen_image)
    



