import re 
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={ids:token for token, ids in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = "".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

