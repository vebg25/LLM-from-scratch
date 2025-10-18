import tiktoken

# Implementation of GPT-2,3 Tokenizer using tiktoken
# Kuch nahi yaar, individual characters ko merge karke subwords create karta hai aur unki vocab banaata hai
tokenizer = tiktoken.get_encoding("gpt2")

# Might create more subwords from a single word but will merge them while decoding 
text="Akwirw ier"
ids = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(ids)
print(tokenizer.decode(ids))