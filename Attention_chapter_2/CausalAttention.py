import torch  
from torch import nn

torch.manual_seed(123)

class CausalAttention(nn.Module):
    """
    Implements Causal (Masked) Self-Attention.
    - Each token can only attend to *itself and previous tokens*.
    - Future tokens are masked out using a triangular attention mask.
    """
    def __init__(self, context_length, dropout, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        
        # Linear projections for Q, K, V
        # Using nn.Linear instead of nn.Parameter improves weight initialization and training stability.
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout applied to attention weights (helps regularization)
        self.dropout = nn.Dropout(dropout)

        # ---- Mask Explanation ----
        # torch.triu creates an upper-triangular matrix with 1s above the diagonal.
        # Example (for context_length=4):
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        #
        # In causal attention, we use this mask to BLOCK attention to *future* tokens.
        # It ensures that token i can only attend to tokens ≤ i.
        #
        # register_buffer() → saves this mask as part of model state (moves with .to(device)), 
        # but it’s NOT a learnable parameter.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x → shape: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # Compute query, key, and value projections
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys    = self.W_key(x)    # (b, num_tokens, d_out)
        values  = self.W_value(x)  # (b, num_tokens, d_out)

        # Compute raw attention scores using QKᵀ
        # Transpose keys along the last two dims to match for matmul
        # Result: (b, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(1, 2)

        # ---- Apply Causal Mask ----
        # mask[:num_tokens, :num_tokens] → trims mask if input sequence < context_length
        # masked_fill_ → replaces masked (True) positions with -inf
        # This makes softmax output ~0 for those positions (effectively ignoring them)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Scale scores by sqrt(d_k) to prevent large dot products (stabilizes softmax)
        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)

        # Apply dropout on attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute the weighted sum of values → context vectors
        context_vectors = attention_weights @ values  # (b, num_tokens, d_out)

        return context_vectors

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
        super().__init__()
        self.heads=nn.ModuleList([CausalAttention(context_length,dropout,d_in,d_out,qkv_bias) for _ in range(num_heads)])
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your (x¹)
    [0.55, 0.87, 0.66],  # journey (x²)
    [0.57, 0.85, 0.64],  # starts (x³)
    [0.22, 0.58, 0.33],  # with (x⁴)
    [0.77, 0.25, 0.10],  # one (x⁵)
    [0.05, 0.80, 0.55]   # step (x⁶)
])

# Stack the same sequence twice → batch of size 2
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # (batch_size=2, seq_len=6, d_in=3)
# ---- Run a forward pass ----
context_length = batch.shape[1]  # seq_len = 6
Attn = CausalAttention(
    context_length=context_length,
    dropout=0.1,
    d_in=batch.shape[-1],
    d_out=2
)

context_vectors = Attn(batch)
print(context_vectors.shape)  # (2, 6, 2)
