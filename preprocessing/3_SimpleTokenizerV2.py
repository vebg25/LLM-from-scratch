import re 

with open('the-verdict.txt','r') as file:
    raw_text=file.read()

    # Step-1 : Converting text into tokens 
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # Step-2 : Creating a vocabulary having each token assigned to an integer
    all_words = sorted(set(preprocessed))
    all_words.extend(["<|UNK|>","<|endoftext|>"])
    vocab = {token:integer for integer,token in enumerate(all_words)}

# Step-3 : Creating a tokenizer for the vocabulary you just created used for encoding and decoding
class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={ids:token for token, ids in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]
        preprocessed = [word if word in self.str_to_int else "<|UNK|>" for word in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    



tokenizer = SimpleTokenizerV2(vocab)



# Utilizing <|UNK|> and <|endoftext|>
text1 = "hello, I am Vaibhav"
text2 = "hello, Mrs. Gisburn said with pardonable"
text = " <|endoftext|> ".join((text1,text2))
print(text)
ids = tokenizer.encode(text)
print(ids)
decoded_text = tokenizer.decode(ids)
print(decoded_text)


# new_ids = obj.encode(new_text)



