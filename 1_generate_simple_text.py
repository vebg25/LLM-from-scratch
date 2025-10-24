from Transformer_Architecture_chapter_3.transformer_block_1 import GPTModel
import torch 
import tiktoken

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

def generate_text_simple(model, idx,max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                   
        probas = torch.softmax(logits, dim=-1)          
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
     
def text_to_token_ids(txt,tokenizer):
    encoded_text = tokenizer.encode(txt,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0) # Adds a batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    decoded = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(decoded.tolist())
    return decoded_text

tokenizer = tiktoken.get_encoding("gpt2")
sample_text = "Every Move takes you a step"

token_ids = generate_text_simple(
    model=model,
    idx = text_to_token_ids(sample_text, tokenizer=tokenizer),
    max_new_tokens = 10,
    context_size= GPT_CONFIG_124M['context_length']
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from preprocessing_chapter_1._6_CreatingDataset import create_dataloader

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)