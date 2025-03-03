import torch

torch.manual_seed(123)#testing purposes
input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dimensions = 3
embedding_layer = torch.nn.Embedding(vocab_size, output_dimensions)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))##print the values for the 4th tensor - > 1