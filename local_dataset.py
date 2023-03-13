import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
def get_dataset(dataset_name):
        
        curr_dataset = {
                'train': pd.read_csv('data/'+dataset_name.lower()+'/train.tsv', sep = '\t'),
                'valid': pd.read_csv('data/'+dataset_name.lower()+'/dev.tsv', sep = '\t'),
                'test': pd.read_csv('data/'+dataset_name.lower()+'/test.tsv', sep = '\t')
               }
        
        curr_dataset['train'] = curr_dataset['train'].replace(np.nan, ' ', regex=True)
        curr_dataset['valid'] = curr_dataset['valid'].replace(np.nan, ' ', regex=True)
        curr_dataset['test'] = curr_dataset['test'].replace(np.nan, ' ', regex=True)
        return curr_dataset

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, device, length = 128, name = 'sst'):
        # sst2, sst1
        self.dataset = dataset
        self.length = length
        self.tokenizer = tokenizer
        self.name = name # name is introduced to allow for adding more datasets other than just sst2, keys vary for each dataset
        self.device = device
    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, idx):
        
        inputs = self.tokenizer(self.dataset['sentence'][idx], padding='max_length', truncation=True, return_tensors="pt", max_length = self.length)
        #print(inputs['input_ids'].size(), inputs['token_type_ids'].size(), inputs['attention_mask'].size(), torch.Tensor([self.dataset[idx]['label']]))
        data_sample =  {
                'input_ids': inputs['input_ids'].squeeze().to(self.device),
                'attention_mask': inputs['attention_mask'].squeeze().to(self.device),
                'labels': torch.Tensor([self.dataset['label'][idx]]).long().to(self.device),
                }
        if 'token_type_ids' in inputs.keys():
            data_sample['token_type_ids'] = inputs['token_type_ids'].squeeze().to(self.device)
        return data_sample
