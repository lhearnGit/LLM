import re
from SimpleTokenizerV1 import SimpleTokenizerV1

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
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

#assign a number to each word
vocab = {token:token_number for token_number, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)
textSample = """"It's the last he painted, you know," Mrs.Gisburn said with pardonable pride."""

encoded = tokenizer.encode(textSample)
print("Encoded Text :", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded Text: ", decoded)


#This will fail and throw a Key Error on "Hello, weather and Glacial", the Tokenizer cannot handle unknowns
#nonSourceTextSample = "Hello, the weather is Glacial" 
#encodeTest = tokenizer.encode(nonSourceTextSample)
#print("Encoded : ", encodeTest)