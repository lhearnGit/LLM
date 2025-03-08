import torch
def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # adds batch dimension [2,4,256]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    # remove batch dimension squishing back down to [4,256]
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

