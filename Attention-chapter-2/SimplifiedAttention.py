import torch 

# Step-1
# Post embedding layer (token embedding + positional encoding)
# Embeddings → semantic + positional info for each token.
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)  
)

# Step-2
# Query = the token we want to focus from (“journey”).
query = inputs[1] # journey word / query word
attn_scores = torch.empty(inputs.shape[0]) # torch.empty(size): This function allocates memory for a tensor of the specified size but does not initialize its contents. The tensor will contain whatever arbitrary data happened to be in that memory location previously. This is often referred to as "uninitialized" or "garbage" values.
for i,x_i in enumerate(inputs):

    # Step-3
    # In attention, the dot product measures how similar or relevant each word is to the query word.
    attn_scores[i]=torch.dot(query,x_i) # Similarity score/attention score calculated using dot product

# Step-4
# Normalization of attention scores such that they sum upto 1 for training stability
# For Normalization, we use Softmax Function
attn_weights = torch.softmax(attn_scores,dim=0)

# Step-5
# For our query vector, we multiply each attention weight with each input and add all the products to create a context vector for the word "journey"

context_vector_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vector_2+=attn_weights[i]*x_i

print(context_vector_2)

# This concludes the simple self attention module for one word, now we have to replicate this for all the word embeddings in the inputs



 