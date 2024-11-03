import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class bert_dataset(Dataset):
    def __init__(self,data_folder):
        super().__init__()
        self.df = pd.read_csv(data_folder)
        return
    
    def __getitem__(self,index):
        return self.df.iloc[index].values 
    
    def __len__(self):
        return self.df.shape[0]

    def show_sample(self,index):
        row = self.__getitem__(index)
        print("Values of a row" , row)
        return row

## Unit testing 

if __name__=="__main__":
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\GenAI_model_implementation\Text_usecase\BERT_embedding_clustering\input_data\spamdata_v2.csv"
    data_obj = bert_dataset(folder)
    print(data_obj.__len__())
    data_obj.show_sample(1)
