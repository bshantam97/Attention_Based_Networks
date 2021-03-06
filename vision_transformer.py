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
import PIL
import math

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

class MLP(nn.Module):

  def __init__(self, in_features, hidden_features = None, out_features = None, drop = 0):
    """
    Multi Layer Perceptron
    Parameters
    ----------
    in_features: Number of input features
    hidden features: Number of hidden features
    out_features: Number of output features
    dropout: Dropout probability

    Attributes
    ----------
    fc1: nn.Linear
        The first linear layer
    activation = nn.GELU
        Gaussian Error Linear Unit
    fc2: nn.Linear
        Second Linear layer
    """

    super(MLP, self).__init__()
    self.fc1 = nn.Linear(in_features = in_features, out_features = hidden_features)
    self.activation = nn.GELU()
    self.fc2 = nn.Linear(in_features = hidden_features, out_features = out_features)
    self.dropout = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.fc2(x) # (batch_size, n_patches+1, out_features)

    return x

class EncoderBlock(nn.Module):

  def __init__(self, embed_dim, heads, mlp_ratio, atten_p = 0, fc_p = 0, qkv_bias = False):
    """Encoder Block
    Parameters
    ----------
    embed_dim:int
      The embedding dimension of the input patches
    heads: int
      Number of head in the transformer encoder
    mlp_ratio: float
      determines the MLP hidden dimension size with respect to embed_dim
    qkv_bias: bool
      If this is true then it includes the bias in query, key and value projections

    Attributes
    ----------

    norm1, norm2: LayerNorm
      Layer Normalization
    mha: Multi head attention
    MLP: The multi layer perceptron module
    """
    super(EncoderBlock, self).__init__()
    self.mha = MultiHeadAttention(
        embed_dim = embed_dim, heads = heads, atten_p = atten_p,
        fc_p = fc_p, qkv_bias = qkv_bias
        )
    # Provides faster training and some amount of regularization
    self.LayerNorm1 = nn.LayerNorm(embed_dim)
    self.LayerNorm2 = nn.LayerNorm(embed_dim)
    hidden_features = int(embed_dim * mlp_ratio)
    self.MLP = MLP(in_features = embed_dim, 
                   hidden_features = hidden_features,
                   out_features = embed_dim)
  def forward(self, x):
    x = x + self.mha(self.LayerNorm1(x))
    x = x + self.MLP(self.LayerNorm2(x))

    return x
    
class VisionTransformer(nn.Module):

  """
  VISION TRANSFORMER

  Parameters
  ----------
  img_size: int
    The input size of the image
  patch_size: int
    The patch size to be extracted from the input image
  in_chans: int
    The number of channels in the input image
  embed_dim: int
    The embedding dimension of the patch input
  n_heads: int
    Number of heads in MultiHead attention
  atten_p, fc_p: float
    dropout probability
  mlp_ratio: float
    Determine the size of the hidden unit
  num_layers: int
    The number of encoder blocks stacked together
  num_classes: int
    The total number of classes for classification  
  Attributes
  ----------
  patch_embed: The Patch Embedding
  cls_token: nn.Parameter
    It is a learnable parameter that will represent the first token in the sequence
  pos_embed: The position Embedding
    Positional embedding for the class token and the image patches
    It has (n_patches+cls_token) * (Embed_dim) elements
  pos_drop: nn.Dropout
    Dropout layer
  """
  def __init__(self, 
               img_size = 384,
               patch_size = 16, 
               in_chans = 3, 
               embed_dim = 768, 
               n_heads = 12,
               num_classes = 1000,
               num_layers = 12,
               qkv_bias = False,
               atten_p = 0,
               fc_p = 0,
               mlp_ratio = 4.0):
    super(VisionTransformer, self).__init__()
    self.embed_dim = embed_dim
    self.layers = nn.ModuleList([
                                 EncoderBlock(
                                     embed_dim = embed_dim,
                                     heads = n_heads,
                                     mlp_ratio = mlp_ratio,
                                     atten_p = atten_p,
                                     fc_p = fc_p,
                                     qkv_bias = qkv_bias
                                 ) for _ in range(num_layers)
    ])
    self.patch_embed = PatchEmbed(img_size = img_size, 
                                  patch_size = patch_size,
                                  in_chans = in_chans,
                                  embed_dim = embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim))
    self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
    
    self.pos_drop = nn.Dropout(p = fc_p)
    self.norm = nn.LayerNorm(embed_dim) # Normalization before the final output pass
    self.head = nn.Linear(embed_dim, num_classes) # For the final output pass

  def forward(self, x):
    """
    The forward pass method
    PARAMETERS
    ----------
    x: torch.Tensor
      This is a 4D input image with size (batch size, in_chans, img_size, img_size)

    RETURNS
    -------
    logits: torch.tensor
      The logits over all the classes (batch_size, num_classes)
    """
    # Create the Patch Embedding
    x = self.patch_embed(x)
    print(self.cls_token.shape)
    # Expand the class tokens along batch dimension
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (n_samples, 1, embed_dim)

    # One can imagine it being something like
    # Concatenation along token dimension
    # Concatenated on top of the embedding

    x = torch.cat((cls_token, x), dim = 1)

    # Add the positional information to the embedding
    x = x + self.pos_embed
    x = self.pos_drop(x)

    for layers in self.layers:
      x = layers(x)
    x = self.norm(x)

    # Hope that this final class embedding that we encoded to the top encodes the information
    # about the image
    final_cls_token = x[:,0]
    x = self.head(final_cls_token)

    return x
# encoder = EncoderBlock(embed_dim = 768, heads = 1, mlp_ratio = 4)
# vision_transformer = VisionTransformer()
# # attn = MultiHeadAttention(embed_dim = 768, heads = 12)
# image = torch.randn((1,3,384,384))
# output = vision_transformer(image)
# output.shape

# encoder = EncoderBlock(embed_dim = 768, heads = 1, mlp_ratio = 4)
# attn = MultiHeadAttention(embed_dim = 768, heads = 12)
# patch_embed = PatchEmbed(img_size = 224, patch_size = 14, in_chans = 3, embed_dim = 768)
# image = torch.randn((3, 3, 224, 224))
# patch = patch_embed(image)
# output1 = encoder(patch)
# output2 = attn(patch)

def patchEmbedding(image_location):
  image = PIL.Image.open(image_location).convert('RGB')
  image_tensor = torchvision.transforms.ToTensor()
  image_transform = torchvision.transforms.CenterCrop(224)
  image_tensor = image_tensor(image)
  image_transform = image_transform(image_tensor)
  image_transform = image_transform.unsqueeze(0) # Add additional dimension as patch embed expects 4D input
  patch_embed = PatchEmbed(img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768)
  patches = patch_embed(image_transform)
  return patches

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

image_dir = '/content/drive/MyDrive/CATS_DOGS/CATS_DOGS/train/CAT/1.jpg'
image_patch_embedding = patchEmbedding(image_location = image_dir)

pe = PositionalEncoding(d_model = 768, max_len = 196)
pe = pe(image_patch_embedding)
image_PIL = torchvision.transforms.ToPILImage()
final_embedding = image_PIL(pe)
plt.imshow(final_embedding)