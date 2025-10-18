import re
with open('the-verdict.txt','r') as file:
    raw_text=file.read()

    # Step-1 : Converting text into tokens 
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # Step-2 : Creating a vocabulary having each token assigned to an integer
    all_words = sorted(set(preprocessed))
    vocab = {token:integer for integer,token in enumerate(all_words)}
    