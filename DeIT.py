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

class MultiHeadAttention(nn.Module):

  def __init__(self, embed_dim, heads, qkv_bias = False, atten_p = 0, fc_p = 0):

    super(MultiHeadAttention, self).__init__()
    self.embed_dim = embed_dim # just for calculation take as 768
    self.heads = heads # Take 12 heads
    self.head_dim = embed_dim // heads # 64-> divided into 12 x 64 parts
    self.scale = self.head_dim ** -0.5 # To divide QK^(T)
    self.atten_p = atten_p # Dropout probability applied to the query, key value vector
    self.fc_p = fc_p # Dropout probability applied to the final fc layer

    assert (self.head_dim * heads == embed_dim)

    # Project the queries key and value h times with different learned linear projections
    # Multiply 3 becaause the output embedding dimensions will be a composition of 
    # Q, K and V
    self.qkv_projection = nn.Linear(self.embed_dim, 3*self.embed_dim, bias = qkv_bias)

    self.attention_dropout = nn.Dropout(atten_p)

    # After Concatenation, pass through a linear layer without changing dimensions
    self.fc_out = nn.Linear(embed_dim, embed_dim) #768->768
    self.fc_drop = nn.Dropout(fc_p)
  
  def forward(self, x):

    """
    Parameters
    ----------
    x: (n_samples, n_patches+1, embed_dim) 
    Basically x is the patch embedding that will be input to the encoder. The one is added
    for the extra learnable class embedding 

    Returns
    -------
    torch.tensor
    Shape: (n_samples, n_patches+1, embed_dim)
    """
    batch_size, n_tokens, embed_dim = x.shape
    qkv = self.qkv_projection(x)
    # Now because we have an image which is a 3D Tensor 5 add 3 channels while reshaping
    qkv = qkv.reshape(
        batch_size, n_tokens, 3, self.heads, self.head_dim
        )
    qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, self.heads,n_patches+1, head_dim)
    queries,keys,values = qkv.chunk(3, dim = 0) #(1, batch_size, self.heads, n_patches+1, head_dim)
    queries,keys,values = queries[0], keys[0], values[0] # (batch_size, self.heads, n_patches+1, head_dim)
    # Now Lets use einsum to compute the softmax
    # Query Shape: (batch_size, self.heads, n_patches+1, head_dim)
    # Key Shape: (batch_size, self.heads, n_patches+1, head_dim)
    # Value Shape: (batch_size, self.heads, n_patches+1, head_dim)
    # QK^(T) Shape = (batch_size, self.heads, n_patches+1, n_patches+1)
    # below b->batch_size, h->number of heads, q-> query length, k->key length, d->head_dim
    qk = torch.einsum('bhqd,bhkd->bhqk',[queries,keys])
    # Take softmax along the key dimension
    attention = torch.softmax(qk/self.scale**(0.5), dim = 3)
    attention = self.attention_dropout(attention)
    weighted_average = torch.einsum('bhqk, bhvd->bqhd',[attention, values]).reshape(
        batch_size, n_tokens, self.embed_dim
    )

    weighted_average_proj = self.fc_out(weighted_average)
    weighted_average_proj = self.fc_drop(weighted_average_proj)

    return weighted_average_proj