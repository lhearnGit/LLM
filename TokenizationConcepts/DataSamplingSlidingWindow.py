import tiktoken
#currently using tiktoken 0.9.0 - 3/3/2025

file_path = "LLM/Data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()


#tiktoken uses Byte-Pair Encodings or BPEs to breakdown input text into subword units then re-assemble
tokenizer = tiktoken.get_encoding("gpt2")

encoded_text = tokenizer.encode(raw_text)#No Allowed Specials such as <|endoftext|>, as there are none in the data sample
print(len(encoded_text))
decoded_text = tokenizer.decode(encoded_text)
print(len(decoded_text))#unlike SimpleTokenizer V2, this BPE tokenizer will handle any possible string value and properly re-assemble it without inserting <|unk|> tokens which result in loss of original text.


encoded_text_sample=encoded_text[:50]

#Basic Visual Demonstration of Sliding Data Window
context_size=4
x = encoded_text_sample[:context_size]
y = encoded_text_sample[:context_size+1]
print(f'x: {x}')
print(f'y:       {y}')

for i in range(1, context_size+1):
    context = encoded_text_sample[:i]#the current context used to predict the next subword unit
    desired = encoded_text_sample[i]#the correct result of a prediction
    print(context,"----->",desired)
    print(tokenizer.decode(context),"----->",tokenizer.decode([desired]))#decoding to plain text to see representation 
    
