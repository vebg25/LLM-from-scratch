import torch 
from torch import nn    
import numpy as np 

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(np.ones(embed_dim))
        self.shift = nn.Parameter(np.zeros(embed_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return (norm_x * self.scale) + self.shift