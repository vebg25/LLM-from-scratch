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
    dataset = create_dataloader(raw_text,max_length=4,batch_size=1,stride=1,shuffle=False)
    iterartor = iter(dataset)
    first_batch = next(iterartor)
    print(first_batch)