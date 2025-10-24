from Transformer_Architecture_chapter_3.transformer_block_1 import GPTModel
import torch 

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,      
    "qkv_bias": False
 }

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

