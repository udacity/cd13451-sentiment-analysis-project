import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class ProductReviewEmbeddings(Dataset):
  r"""RoBERTa embeddings of customer reviews. Embeddings are precomputed 
  and saved to disk. This class does not compute embeddings live.

  Argument
  --------
  split (str): the dataset portion
    Options - train | dev | test | unlabeled | * 
    If unlabeled, the __getitem__ function will set `label` to -1.
  """

  def __init__(self, split='train'):
    super().__init__()
    
    if split == 'train':
        train_data_dir = os.environ['SM_CHANNEL_TRAIN']
        self.data = pd.read_csv(os.path.join(train_data_dir, 'train.csv'))
        self.embedding = torch.load(os.path.join(train_data_dir, f'{split}.pt'))
    elif split == 'dev':
        dev_data_dir = os.environ['SM_CHANNEL_TRAIN']
        self.data = pd.read_csv(os.path.join(dev_data_dir, 'dev.csv'))
        self.embedding = torch.load(os.path.join(dev_data_dir, f'{split}.pt'))
    else:
        test_data_dir = os.environ['SM_CHANNEL_TRAIN']
        self.data = pd.read_csv(os.path.join(test_data_dir, 'test.csv'))
        self.embedding = torch.load(os.path.join(test_data_dir, f'{split}.pt'))
    
    self.split = split

  def __getitem__(self, index):
    inputs = self.embedding[index].float()
    label = int(self.data.iloc[index].label)
    return inputs, label

  def __len__(self):
    return len(self.data)