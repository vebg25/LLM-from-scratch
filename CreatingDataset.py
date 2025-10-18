import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken
class DatasetV1:
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]

        token_ids = tokenizer.encode(txt)
        length=len(token_ids)
        for i in range(0,length-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader(txt,max_length=256,batch_size=1,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = DatasetV1(txt=txt,tokenizer=tokenizer,max_length=max_length,stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

with open('the-verdict.txt','r') as file:
    raw_text = file.read()

    vocab_size = 50257
    output_dim = 256
    max_length=4
    dataset = create_dataloader(raw_text,max_length=max_length,batch_size=8,stride=max_length,shuffle=False)
    iterartor = iter(dataset)
    inputs,target = next(iterartor)

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # Embedding layer for token id encoding having vocab_size as input and output_dim as output
    token_embeddings = token_embedding_layer(inputs)

    context_length=max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = pos_embeddings+token_embeddings