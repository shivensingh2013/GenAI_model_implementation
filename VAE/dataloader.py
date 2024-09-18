
import torch
from  torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class custom_dataset(Dataset):
    def __init__(self, data_folder ):
        super().__init__()
        ## read the image data here
        self.df = pd.read_csv(data_folder)
        return 
    
    def __getitem__(self,index):
        return self.df.iloc[index].values 

    def __len__(self):
        return self.df.shape[0]

    def show_sample(self,index):
        row = self.__getitem__(index)
        img_array = np.array(row[1:],dtype = int)
        img_array = np.reshape(img_array, (28,28))
        img = Image.fromarray(np.uint8(img_array ) , 'L')
        img.save("sample.png")
        return img

if __name__ == "__main__":
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\fashion_mnist\fashion-mnist_train.csv"
    dataset_obj = custom_dataset(folder)
    # print(dataset_obj.__len__())
    # print(dataset_obj.__getitem__(2))
    # dataset_obj.show_sample(1)













