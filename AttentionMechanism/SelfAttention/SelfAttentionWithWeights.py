import torch

starterText = "Your journey starts with one step"
inputs = torch.tensor([
    [ .43 , .15 , .89], #Your
    [ .55, .87 , .66], #journey
    [ .57 , .85 , .64], #starts
    [ .22 , .58 , .33], #with
    [ .77 , .25 , .10], #one
    [ .05 , .80 , .55], #step
])
torch.manual_seed(123)


## Calculating One Context Vector
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 ## sets output dimension

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 



query_2 = x_2 @ W_query ##Tensor Input Values of Journey @ Randomized_initial_Weights Matrix
key_2 = x_2 @ W_key ##Tensor Input Values of Journey @ Randomized_initial_Weights Matrix
value_2 = x_2 @ W_value ## Tensor Input Values of Journey @ randomized_initial_Values matrix

print(query_2)


keys = inputs @ W_key ## All inputs @ All Respective Keys
values = inputs @ W_value ## all Inputs @ all Respective Values

print("Keys.shape ",keys.shape)
print("values.shape" , values.shape)

keys_2 = keys[1] ##

attn_score_22 = query_2.dot(keys_2)## Scalar Dot Multiplier, 
print("Attention_Score : ",attn_score_22) 

attention_scores2 = query_2 @ keys.T
print("Scores : ",attention_scores2)

d_k = keys.shape[-1]

##d_k**0.5 = take the square_root of the embedding dimension of the keys
attn_weights_2 = torch.softmax(attention_scores2 / d_k**0.5, dim=-1) 

print("Weights: ", attn_weights_2)

context_vectors_2 = attn_weights_2 @ values

print("context",context_vectors_2)