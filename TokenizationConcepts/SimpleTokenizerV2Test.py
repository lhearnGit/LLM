import re
from SimpleTokenizerV2 import SimpleTokenizerV2

file_path = "LLM/Data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total Number of Characters ", len(raw_text))
print(raw_text[:99])

#Parse Input Text into words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

#Sort Words into set, and create a vocabulary list
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>","<|unk|>"])#Add an End Of Text Token to allow delimiting of batches, and an UNK token to replace unknown words
vocab_size = len(all_tokens)
print(vocab_size)

#assign a number to each word
vocab = {token:token_number for token_number, token in enumerate(all_tokens)}

tokenizer = SimpleTokenizerV2(vocab)
textSample = """"It's the last he painted, you know," Mrs.Gisburn said with pardonable pride."""

encoded = tokenizer.encode(textSample)
print("Encoded Text :", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded Text: ", decoded)


#unlike V2 this will succeed, Hello, weather and Glacial are replaced with <|unk|>
nonSourceTextSample = "Hello, the weather is Glacial" 
encodeTest = tokenizer.encode(nonSourceTextSample)
decodeTest = tokenizer.decode(encodeTest)
print("Encoded : ", encodeTest)
print("Decoded : ", decodeTest)


#should out put <|unk|> , the <|unk|> is <|unk|> <|endoftext|> sample2....
multiSample1 ="Hello, the Weather is Glacial"
multiSample2 ="Yes, it is rather Frigid today."
multiTextBatchSample = " <|endoftext|> ".join((multiSample1, multiSample2))

encodeTest2 = tokenizer.encode(multiTextBatchSample)
decodeTest2 = tokenizer.decode(encodeTest2)
print("Encoded : ", encodeTest2)
print("Decoded : ", decodeTest2)



