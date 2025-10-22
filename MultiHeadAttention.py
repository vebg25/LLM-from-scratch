import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout , qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_query = nn.Linear(d_in,d_out, qkv_bias=False)
        self.W_key = nn.Linear(d_in,d_out, qkv_bias=False)
        self.W_value = nn.Linear(d_in,d_out,qkv_bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out,d_out)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attention_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]

        attention_scores.masked_fill_(mask_bool)

        dim_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores/(dim_k**0.5),dim=-1)

        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(1,2) 
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors