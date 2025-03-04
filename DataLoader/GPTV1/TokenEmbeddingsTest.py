import torch
from CreateDataLoaderV1 import create_data_loader

torch.manual_seed(123)
vocab_size = 50257  #number of tokens in tiktoken
output_dimensions = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dimensions)

with open("../../Data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
context_length = max_length

dataloader = create_data_loader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iterator = iter(dataloader)
inputs, targets = next(data_iterator)

print(inputs)
print("InputShape\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

pos_embedding_layer = torch.nn.Embedding(context_length, output_dimensions)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
