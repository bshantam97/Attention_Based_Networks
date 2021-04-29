import os 
import time 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.tensorboard import SummaryWriter

class PatchEmbed(nn.Module):
  """Splits the image into patches and then embeds them

  Parameters
  ----------
  img_size : Size of the image (Its a square)
  patch_size: Size of the patch
  in_chans: Number of input channels
  embed_dim: The embedding dimension

  Attributes
  ----------

  n_patches: Number of patches inside our image
  proj: nn.Conv2d Convolution that performs the splitting and embedding
  """

  def __init__(self, img_size, patch_size, in_chans = 3, embed_dim = 768):

    super(PatchEmbed, self).__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.in_chans = in_chans
    self.n_patches = (img_size // patch_size) ** 2
    self.embed_dim = embed_dim #{P^(2)C}

    # Divide into patches
    self.proj = nn.Conv2d(
        in_chans, 
        embed_dim,
        kernel_size = patch_size,
        stride = patch_size
    )
  
  def forward(self, x):
    """
    Run forward pass

    Parameters
    ----------
    x: torch.tensor(n_samples, in_chans, img_size, img_size) # (C,H,W)

    Returns
    -------
    torch.tensor (n_samples, num_patches, embed_dim) # (num_patches)x(Patch^(2){Channels})
    
    Intuition:
    Imagine we have an input image with batch size of 3 , 3 channels and height and width dimensions
    of 48. Now we select our patch size as 16 and apply a kernel of shape (3,16,16). This would give
    the output dimension as (3,768,3,3). We selected the output number of channels as 768 which is the 
    embedding dimension or P^(2)C = 16^(2){(3)}. Now we flatten it along the height and width dimension
    to obtain an output size of (3,768,9). The final step is to transpose along the index 1 and index 2
    to obtain the final embedding as (3,9,768). Now one can relate this as to how one forms word embeddings
    in NLP. If we have 9 words we pick each word, choose the embedding dimension , create a 1D embedding 
    and concatenate it and then pass it as an input into the transformer encoder. In the case of images there comes 
    a third dimension for the channels.    
    """
    x = self.proj(x) #(n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
    x = x.flatten(2) # (n_samples, embed_dim, n_patches) , Multiples the last 2 dimensions
    x = x.transpose(1,2) # (n_samples, n_patches, embed_dim) # transposes the index 1 with index 2

    return x