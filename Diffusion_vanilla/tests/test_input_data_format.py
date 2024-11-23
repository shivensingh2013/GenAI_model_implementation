import pytest
from config import CONFIG

from dataset import diffusion_dataset
@pytest.fixture()
def input_data():
    cfg = CONFIG()
    data_obj = diffusion_dataset(cfg.train_csv_path)
    return  data_obj
    
def test_inpuit_len():
    ## Given - train data size is 60000
    ## When data is loaded 
    cfg = CONFIG()
    len_dataset = diffusion_dataset(cfg.train_csv_path).__len__()
    len_test  = diffusion_dataset(cfg.test_csv_path).__len__()
    ## Then 
    assert len_dataset == 60000
    assert len_test == 10000

def test_data_shape(input_data):
    sample = input_data.__getitem__(2)
    assert list(sample.size())== [1,28,28]


