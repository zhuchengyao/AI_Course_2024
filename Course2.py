import torch
from torch import nn

vocab_size = 16000
seq_len = 128
d_model = 128
n_layer = 8
n_head = 4

class Sinusiod



class GPTModel(nn.Module):

    def __init__(self):
        self.tok_embed_table = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        pass
