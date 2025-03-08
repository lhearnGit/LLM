import torch
import tiktoken
from GPT2Model import GPT2Model
from GPT2Parts.GenerateTextSimple import generate_text_simple

def text_to_token(text,tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # adds batch dimension [2,4,256]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    # remove batch dimension squishing back down to [4,256]
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M = {
    "vocab_size":50257, #Vocab Size, number of BPE Tokens
    "context_length":256, #minimum context block size for gpt2
    "emb_dim":768, #minimum embed dims for gpt2
    "n_heads":12, 
    "n_layers":12,
    "drop_rate":.1,
    "qkv_bias":False
}

file_path = "LLM/Data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
model = GPT2Model(GPT_CONFIG_124M)
model.eval()


# Validate File Loads Properly
# print(text_data[:99])


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print('Character count : ' , total_characters)
print('Token Count : ' , total_tokens)

text1 = "Every effort moves you"
text2 = "I really like chocolate"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token(text1,tokenizer=tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)


inputs = torch.tensor([
    [16833, 3626, 6100], # Every effort moves
    [40, 1107, 558] # I really like
    ])

# shift elements +1 
targets = torch.tensor([
    [3626,6100,345], #effort moves you
    [1107,588,11311] #really like chocolate
    ])


# Training Steps  - Performing Back Propagation - Loss function
# Manual Method ------------------------------------------------------------
# Obtain Logits 
#  - > Calculate Probabilities
#       - > Obtain Target Probabilities
#           - > Obtain Log Probabilities
#               - > Obtain Log Average Probabilities
#                   - > Obtain Log Average Probabilities
#                       Goal - > Obtain Negative Log Average Probabilities


# Obtain Logits
with torch.no_grad():
    logits = model(inputs)
# Calc Probabilities
probas = torch.softmax(logits, dim=-1)
print(probas.shape)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(token_ids)
print(f'batch 1 target {token_ids_to_text(targets[0], tokenizer=tokenizer)}' )
print(f'batch 1 Outputs {token_ids_to_text(token_ids=token_ids[0].flatten(), tokenizer=tokenizer)}' )


# Calc Target Probas
text_idx = 0
target_probas_1 = probas[text_idx , [0,1,2], targets[text_idx]]

print(f'Text {text_idx+1}', target_probas_1 )
text_idx=1
target_probas_2 = probas[text_idx , [0,1,2], targets[text_idx]]
print(f'Text {text_idx+1}', target_probas_2 )


# Calc Log Probas

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# Calc Log Avg Probas
avg_log_probs = torch.mean(log_probas)
print(avg_log_probs)
# Calc Neg Avg Probas
# turning negative value positive - Cross Entropy Loss
neg_avg_log_probs = avg_log_probs * -1
print(neg_avg_log_probs)

# Cross Entropy - essentially the same process
# PyTorch can perform the entire process via cross_entropy 

print("Logits Shape", logits.shape)
print("Targets Shape", targets.shape)
# Flatten to [2,3] across the batch dimension
logits_flat = logits.flatten(0,1) 

targets_flat = targets.flatten()

print("Logits Flattened", logits_flat.shape)
print("Targets Flattened", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat,targets_flat)
print('loss',loss)
print(neg_avg_log_probs)

#Check that manual method is infact equal to the cross_entropy method
if loss == neg_avg_log_probs:
    print(f'{loss} equals {neg_avg_log_probs}')
else:
    print(f'{loss} does not  {neg_avg_log_probs}')

# Perplexity Values
# Perplexity is a measurement of uncertainty
# a Perplexity value of 25,000 means the model is unsure of which of the 25,000 tokens the next value should relate too

perplexity = torch.exp(loss)

print(perplexity)