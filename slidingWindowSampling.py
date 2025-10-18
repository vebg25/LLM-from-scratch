import tiktoken

with open("the-verdict.txt","r") as file:
    text = file.read()
    tokenizer = tiktoken.get_encoding("gpt2")

    encoded_text = tokenizer.encode(text)

    enc_sample = encoded_text[50:]
    context_size=4
    for i in range(1,context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(f'{tokenizer.decode(context)} ---> {tokenizer.decode([desired])}')
