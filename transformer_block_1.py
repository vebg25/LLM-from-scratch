from Attention_chapter_2.MultiHeadAttention import MultiHeadAttention
import torch 
from torch import nn 
from llm_architecture_1 import GPT_CONFIG_124M
from LayerNormalization import LayerNorm
from GELU import GELU, FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg)
        self.norm_1 = LayerNorm(cfg['emb_dim'])
        self.norm_2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x 
        x = self.norm_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        return x 
    
torch.manual_seed(123)
model = TransformerBlock(GPT_CONFIG_124M)
data = torch.rand(2,4,768)

output = model(data)

print(output.shape)