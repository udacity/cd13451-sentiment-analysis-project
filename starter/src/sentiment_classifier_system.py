r"""A PyTorch Lightning system for training a sentiment classifier."""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pytorch_lightning as pl

class SentimentClassifierSystem(pl.LightningModule):
  """A Pytorch Lightning system to train a model to classify sentiment of 
  product reviews. 

  Arguments
  ---------
  args: input parameters
  """
  def __init__(self, args, callbacks):
    super().__init__()
#     self.save_hyperparameters(args) is used to save the hyperparameters. PyTorch Lightning allows you to access these     
#     hyperparameters via self.hparams.
    self.save_hyperparameters(args)

    # load model
    self.model = self.get_model()

    # We will overwrite this once we run `test()`
    self.test_results = {}
    
    self.model_checkpoint = callbacks[0]

  def get_model(self):
    model = nn.Sequential(
      nn.Linear(768, self.hparams.model_width),
      nn.ReLU(),
      nn.Linear(self.hparams.model_width, 1),
    )
    return model
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), 
      lr=self.hparams.lr)
    return optimizer

  def _common_step(self, batch, _):
    """
    Arguments
    ---------
    embs (torch.Tensor): embeddings of review text
      shape: batch_size x 768
    labels (torch.LongTensor): binary labels (0 or 1)
      shape: batch_size
    """
    embs, labels = batch

    # forward pass using the model
    logits = self.model(embs)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      num_correct = torch.sum(preds.squeeze() == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy

  def training_step(self, train_batch, batch_idx):
    loss, acc = self._common_step(train_batch, batch_idx)
    self.log_dict({'train_loss': loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, dev_batch, batch_idx):
    loss, acc = self._common_step(dev_batch, batch_idx)
    return loss, acc

  def validation_epoch_end(self, outputs):
    avg_loss = torch.mean(torch.stack([o[0] for o in outputs]))
    avg_acc = torch.mean(torch.stack([o[1] for o in outputs]))
    self.log_dict({'dev_loss': avg_loss, 'dev_acc': avg_acc},
      on_step=False, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, test_batch, batch_idx):
    loss, acc = self._common_step(test_batch, batch_idx)
    return loss, acc

  def test_epoch_end(self, outputs):
    avg_loss = torch.mean(torch.stack([o[0] for o in outputs]))
    avg_acc = torch.mean(torch.stack([o[1] for o in outputs]))
    # We don't log here because we might use multiple test dataloaders
    # and this causes an issue in logging
    results = {'loss': avg_loss.item(), 'acc': avg_acc.item()}
    # HACK: https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    self.test_results = results

  def predict_step(self, batch, _):
    logits = self.model(batch[0])
    probs = torch.sigmoid(logits)
    return probs
