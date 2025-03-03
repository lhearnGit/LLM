import re

#Parse Input Text into words

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
for i, item in enumerate(vocab.items()):
    print(item)
    if i == 50: 
        break

