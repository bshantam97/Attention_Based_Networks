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

    # heads: Number of parts to split embed. If we have embed size 256 split into 8x32 parts
    def __init__(self, embed_size, heads):

        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # 256
        self.heads = heads # 8 
        self.head_dim = embed_size // heads # 32

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by head"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False) # 32->32
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False) # 32->32
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False) # 32->32
        
        # After Concatenation
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) # 256 -> 256
    
    def forward(self, values, keys, query, mask):

        # Number of training examples
        N = query.shape[0]

        # 1*x, Count the number of columns to compute the length
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces (batch_size, sequence_length, embedding)->(batch_size, sequence_length, number_heads, head_dim)
        # Number of heads is in the channel dimension(Just for sake of intuition)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, head, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):

        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

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
        out = self.dropout(self.norm2(forward + x))
        return out