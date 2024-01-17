from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.product_review_embeddings import ProductReviewEmbeddings

class ReviewDataModule(pl.LightningDataModule):
  r"""Data module wrapper around review datasets."""

  def __init__(self, args):
    super().__init__()

    train_dataset = ProductReviewEmbeddings(split='train')
    dev_dataset = ProductReviewEmbeddings(split='dev')
    test_dataset = ProductReviewEmbeddings(split='test')

    self.train_dataset = train_dataset
    self.dev_dataset = dev_dataset
    self.test_dataset = test_dataset
    self.batch_size = args.batch_size

  def train_dataloader(self):
    # Create a dataloader for train dataset. 
    # Notice we set `shuffle=True` for the training dataset.
    return DataLoader(self.train_dataset, batch_size = self.batch_size,
      shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.dev_dataset, batch_size = self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size)