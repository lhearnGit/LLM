import tiktoken
#currently using tiktoken 0.9.0 - 3/3/2025
multiSample1 ="Hello, the Weather is Glacial"
multiSample2 ="Yes, it is rather Frigid today."
multiTextBatchSample = " <|endoftext|> ".join((multiSample1, multiSample2))

#tiktoken uses Byte-Pair Encodings or BPEs to breakdown input text into subword units then re-assemble
tokenizer = tiktoken.get_encoding("gpt2")

integers = tokenizer.encode(multiTextBatchSample, allowed_special={"<|endoftext|>"})#allow special characters 
print(integers)
strings = tokenizer.decode(integers)
print(strings)#unlike SimpleTokenizer V2, this BPE tokenizer will handle any possible string value and properly re-assemble it without inserting <|unk|> tokens which result in loss of original text.
