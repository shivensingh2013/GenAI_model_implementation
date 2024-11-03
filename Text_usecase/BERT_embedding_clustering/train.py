## training a simple BERT based attention model
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from dataset import bert_dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader


# specify GPU
device = torch.device("cuda")

if __name__ == "__main__":
    ## loading sentence dataset
    folder = r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\GenAI_model_implementation\Text_usecase\BERT_embedding_clustering\input_data\spamdata_v2.csv"
    dataset= bert_dataset(folder)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size-val_size

    train_dataset, val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    train_dataloader = DataLoader(dataset =train_dataset,batch_size = 8,shuffle = True ) 
    val_dataloader = DataLoader(dataset =val_dataset,batch_size = 8,shuffle = True )
    test_dataloader = DataLoader(dataset =test_dataset,batch_size = 8,shuffle = True )
    print(train_dataloader.__len__(),test_dataloader.__len__(),val_dataloader.__len__())


    ## preparing input for BERT
    """every input embedding is a combination of 3 embeddings-
    1) Vector of input - tokenized data is embedded as a vector
    2) Segment embedding
    3) positinal embedding
    """

    ## loading model - BERT-Base 

    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    ## 