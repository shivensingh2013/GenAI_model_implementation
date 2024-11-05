import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
class diffusion_dataset(Dataset):
    def __init__(self,data_folder,num_datapoints = None):
        super().__init__()
        self.df = pd.read_csv(data_folder)
        if num_datapoints is not None:
            self.df = self.df.iloc[0:num_datapoints]
        return
    
    def __getitem__(self,index):
        img = self.df.iloc[index].filter(regex='pixel').values
        img =  np.reshape(img, (28, 28)).astype(np.uint8)

        # Convert to Tensor
        img_tensor = torchvision.transforms.ToTensor()(img) 
        img_tensor = 2*img_tensor - 1 # [-1, 1]

        return img_tensor
    
    def __len__(self):
        return self.df.shape[0]

    def show_sample(self,index):
        row = self.__getitem__(index)
        img_array = np.array(row[1:],dtype = int)
        img_array = np.reshape(img_array, (28,28))
        img = Image.fromarray(np.uint8(img_array ) , 'L')
        img.save("sample.png")
        return img

## Unit testing 

if __name__=="__main__":
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\GenAI_model_implementation\Diffusion_vanilla\dataset\fashion-mnist_test.csv"
    data_obj = diffusion_dataset(folder)
    transforms.ToPILImage()(data_obj.__getitem__(5)).show()
