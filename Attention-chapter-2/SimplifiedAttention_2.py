# Same as SimplifiedAttention but in a lot less manual coding and use of matrix multiplications to speed up operations
import torch 

inputs = torch.tensor([[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)  
)

attention_scores = inputs @ inputs.T # Dot product
attention_weights = torch.softmax(attention_scores,dim=-1) # Normalization
print(attention_weights.sum(dim=-1))

context_vectors = attention_weights @ inputs # Weighted sum of product of normalized attention scores and inputs to produce context vectors
print(context_vectors)
