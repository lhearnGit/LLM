import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):#Max Length - The number of tokens in a row - Size of Batch of Text, Stride how far the "Selection" moves per step 
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]#get a chunk of words from index i to i+max_length eg. i==0 max_range = 5 input_chunk=[:5]
            target_chunk = token_ids[i+1:i + max_length +1]#get the next chunk of data one index further after [a,b,c] -> target_chunk[d]
            self.input_ids.append(torch.tensor(input_chunk))#return and store a tensor holding the input chunk and target_chunks
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

