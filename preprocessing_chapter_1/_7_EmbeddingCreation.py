import torch 
# If vocabulary is having only 6 words and embedding dimensions should have 3, then this is how we intialize and embedding layer
vocab_size = 6 
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)