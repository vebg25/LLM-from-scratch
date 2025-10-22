import torch 

inputs = torch.tensor([[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)  
)

torch.manual_seed(123)

x2=inputs[1]
d_in = inputs.shape[1]
d_out=2


# torch.rand() creates a tensor having the shape mentioned with each value ranging from 0 to 1
W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)

# Creating query, key and value vectors for our embedding input
query_x2 = x2 @ W_query
key_x2 = x2 @ W_key
value_x2 =x2 @ W_value

# Creating keys and values for all inputs 
keys = inputs @ W_key
values = inputs @ W_value

# Calculate Attention scores for query ("journey" embedding)
attention_scores = query_x2 @ keys.T

# Normalization of attention scores by dividing them by the embedding dimension of keys
dim_k=keys.shape[-1]
attention_weights_2 = attention_scores/(dim_k**0.5)

# Applying softmax function
attention_weights_2 = torch.softmax(attention_weights_2, dim=-1)

# Calculating context vectors from attention weights using values vector
context_vectors = attention_weights_2 @ values
print(context_vectors)