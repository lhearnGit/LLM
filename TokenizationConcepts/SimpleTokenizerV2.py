import re

#Token <|unk|> must be added to vocab arg to handle unknown tokens
#Token <|endoftext> should be added to handle separation of text batches
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {number:string for string, number in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [ item if item in self.str_to_int else "<|unk|>" for item in preprocessed] #add <|unk|> for any word not in the vocabulary
        ids = [self.str_to_int[string] for string in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[number] for number in ids])
        text = re.sub(r'\s+([,.:;?_!"()\']|--|\s])',r'\1', text)
        return text
