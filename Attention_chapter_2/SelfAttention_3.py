''' This is same as SelfAttention_2.py but only replaces torch.nn.Parameter with torch.nn.Linear for better weight initialization and stabilized training '''
import torch
from torch import nn

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in,d_out,bias=qkv_bias) # Changes here
        self.W_key = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self,x):
        query = self.W_query(x) # Changes here
        keys = self.W_key(x)
        values = self.W_value(x)
        attention_scores = query @ keys.T
        dim_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores/(dim_k**0.5),dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors


torch.manual_seed(123)
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)  
)

attn = SelfAttentionV1(inputs.shape[-1],2)
context_vectors = attn(inputs)
print(context_vectors)