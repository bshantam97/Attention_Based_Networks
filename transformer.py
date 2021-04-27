import os 
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.functional as F

parser = argparse.ArgumentParser()
args = vars(parser.parse_args())

class SelfAttention(nn.Module):

  # Here heads are the number of parts to split the embedding. If we have an 
  # Embedding size of 512 split it into 8x64 parts
  def __init__(self, embed_size, heads):

    super(SelfAttention, self).__init__()
    self.embed_size = embed_size # 512
    self.heads = heads # 8
    self.head_dim = embed_size // heads # 64

    #Embed size needs to be divisible by head
    assert (self.head_dim * heads == embed_size) 

    # Project Queries, Keys and Values h times with different learned linear projections
    # to d_k, d_k and d_v dimensions
    self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)

    # After Concatenation
    self.fc = nn.Linear(heads*self.head_dim, embed_size) # 256->256
  
  def forward(self, values, keys, query, mask):
    
    # Number of training examples (Batch Size)
    N = query.shape[0]

    #1*x, Count the number of columns to calculate the length
    # Can be considered as d_model
    # Source and target sentence Length
    # SHAPE: (Batch, Seq_len, Embedding_Dim)
    # Seq_len is the length of the input sentence
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # Split the embedding into self.heads pieces (Batch_Size, Seq_len, embedding)->(batch, seq_len, num_heads, head_dim)
    # Number of heads is in the channel dimension
    # Sequence length can be thought of as the number of words in a sentence
    # Lets say we have an output embedding for 
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, query_len, self.heads, self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)
    # Computer the Dot product : (Q.K)
    # Will be used to compute Softmax: (Softmax(QK^(T)) / root(d_k))V
    # Queries shape = (N, query_len, heads, head_dim)
    # Keys Shape = (N, key_len, head, head_dim)
    # Energy Shape = (N, heads, query_len, key_len)
    # torch.bmm-> Batch Matrix Multiply
    energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

    # ADD MASKING
    # Just for intuition lets say we have a matrix [[1,2,3],[4,5,6]]
    # Now we want to change 1 and 4 to 9.9. Instead of writing a function
    # We could create a mask [[0,1,1],[1,0,1]]. After creating this mask we could
    # Call the masked_fill function like A.masked_fill(mask = 0, 9.9)
    # What this is esentially saying is that for each each 0 position in the mask
    # Change it to 9.9 in the original matrix
    # Similarly below because we dont want QK^TV to peek ahead We could create a mask and fill the future values with -infinity
    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))
    
    # Apply the Softmax function
    # Normalize along the Key Length
    attention = torch.softmax(energy / (self.embed_size**(1/2)), dim = 3)

    # ATTENTION SHAPE: (N, heads, query_len, key_len)
    # VALUES_SHAPE: (N, value_len, heads, heads_dim)
    # Final Dimension: (N, query_len, heads, head_dim)
    # Concatenate the final outputs. Can just reshape (8,64)->(512)
    output = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(
        N, query_len, self.embed_size
        )
    # Send through fully connected layer
    fc_out = self.fc(output)
    return fc_out

class TransformerBlock(nn.Module):

  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)
    # This is a position wise feed-forward network
    # FFN(x) = max (0, xW1 + b1)W2 + b2
    # The inner Feed forward network has a dimensionality of 2048 and the output has a dimensionality of 512
    # The MLP adds an extra layer of complexity and is added to each position identically.
    # This could be thought of as a post process operation to prepare for the next attention block

    self.feed_forward = nn.Sequential(
        nn.Linear(embed_size, forward_expansion*embed_size),
        nn.ReLU(),
        nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)
    x = self.dropout(self.norm1(attention+query))
    forward = self.feed_forward(x) 
    out = self.dropout(self.norm2(forward+x))
    return out

class Encoder(nn.Module):
  def __init__(self,
              src_vocab_size,
              embed_size,
              num_layers,
              heads,
              device,
              forward_expansion,
              dropout,
              max_length):
    super(Encoder, self).__init__()

    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    # ModuleList allows one to store the Module as a list.
    # It can be useful when you need to iterate through layer and store/use some information, like in U-net.

    self.layers = nn.ModuleList([
                             TransformerBlock(embed_size, heads, 
                                              dropout = dropout, 
                                              forward_expansion = forward_expansion) 
                             for _ in range(num_layers)
                             ]
                             )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, mask):

    N, seq_length = x.shape

    # Expand for every example
    positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)
    out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
    for layer in self.layers:
      # We write out out out to indicate query key and mask
      # One can think of it as saving intermediate outputs and passing it to 
      # Future Encoder Blocks
      # ModuleList is benefitial here. If we have 8 Encoders the output of each encoder
      # Is passed as input to the next encoder.
      out = layer(out,out,out,mask)

    return out

class DecoderBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout, device):
    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
        embed_size, heads, dropout, forward_expansion
    )
    self.dropout = nn.Dropout(dropout)
  
  # src_mask is optional
  def forward(self, x, value, key, src_mask, target_mask):
    attention = self.attention(x,x,x,target_mask)
    query = self.dropout(self.norm(attention+x))
    out = self.transformer_block(value, key, query, src_mask)
    return out

class Decoder(nn.Module):

  def __init__(self, 
               target_vocab_size,
               embed_size,
               num_layers,
               heads,
               forward_expansion,
               dropout,
               device,
               max_length):
    
    super(Decoder, self).__init__()
    self.device = device
    self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)
    # Iteration
    self.layers = nn.ModuleList([
                               DecoderBlock(embed_size, heads, forward_expansion, dropout, device) 
                               for _ in range (num_layers)
    ])

    self.fc_out = nn.Linear(embed_size, target_vocab_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, enc_out, src_mask, target_mask):

    # For NLP Imagine that you have a sequence of words of shape (Batch Size, Sequence Length)
    N, seq_length = x.shape
    positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)
    x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
    for layer in self.layers:
      x = layer(x, enc_out, enc_out, src_mask, target_mask)
    out = self.fc_out(x)
    return out

class Transformer(nn.Module):

  # The padding will be used to generate the mask
  def __init__(self,
               src_vocab_size,
               target_vocab_size,
               src_pad_idx,
               target_pad_idx,
               embed_size = 256,
               num_layers = 6,
               forward_expansion = 4,
               heads = 8,
               dropout = 0,
               device = "cuda",
               max_length = 100):
    
    super(Transformer, self).__init__()
    self.encoder = Encoder(src_vocab_size,
                           embed_size,
                           num_layers,
                           heads,
                           device,
                           forward_expansion,
                           dropout,
                           max_length)
    self.decoder = Decoder(
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length)
    
    self.src_pad_idx = src_pad_idx
    self.target_pad_idx = target_pad_idx
    self.device = device
  
  def make_src_mask(self, src):

    # Now the input array is formed by stacking sequences of different lengths together
    # They are post padded with 0's to get the same length
    # Want the Neural Network to ignore these values. Here the padding mask can help
    # It first converts the 0's to 1's and non-zero values to 0's and then multiply with a large negative value
    # After adding the mask to the scaled score the softmax is going to ignore the large negative values at 
    # the empty locations

    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(self.device)
  
  def make_target_mask(self, target):
    N, target_len = target.shape

    # Lower triangular Matrix
    # Basically if we just use torch.tril(torch.ones(5,5)) we will get a 
    # 5x5 matrix filled with 1's in the lower triangle
    # when we use the expand function in this example lets say N = 5 target_len = 5
    # Then for a batch size of 5 for each example in that batch we will have a 
    # (1,5,5) tensor. VERY IMPORTANT TO REMEMBER. Use the expand function frequently

    target_mask = torch.tril(torch.ones(target_len, target_len)).expand(
        N, 1, target_len, target_len
    )
    return target_mask.to(self.device)
  
  def forward(self, src, target):

    src_mask = self.make_src_mask(src)
    target_mask = self.make_target_mask(target)
    enc_src = self.encoder(src, src_mask)
    out = self.decoder(target, enc_src, src_mask, target_mask)

    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
trg = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device = device).to(device)

# Shift the target by 1 so that it doesnot have end of sentence token
out = model(x, trg[:, :-1])
print(out.shape)