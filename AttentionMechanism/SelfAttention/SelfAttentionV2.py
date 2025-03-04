import torch.nn as nn
import torch


inputs = torch.tensor([
    [ .43 , .15 , .89], #Your
    [ .55, .87 , .66], #journey
    [ .57 , .85 , .64], #starts
    [ .22 , .58 , .33], #with
    [ .77 , .25 , .10], #one
    [ .05 , .80 , .55], #step
])

x_2 = inputs[1]
d_in = inputs.shape[1] ##embedding size d=3
d_out = 2 ## sets output dimension

class SelfAttention_V2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        ##Initialize Matricies with randomized values, based on input & output dimension sizes
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys =  self.W_key(x) # using Linear, pass in the input tensor 
        queries =  self.W_query(x)
        values =  self.W_value(x)

        #Calculate the attention scores for the input x,
        attn_scores = queries @ keys.T # Omega Values
        #perform softmax calculation to reduce down to sums of 1
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
    
    def getWeights(self, x):
        keys =  self.W_key(x) # using Linear, pass in the input tensor 
        queries =  self.W_query(x)
        values =  self.W_value(x)

        #Calculate the attention scores for the input x,
        attn_scores = queries @ keys.T # Omega Values
        #perform softmax calculation to reduce down to sums of 1
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        return attn_weights


torch.manual_seed(789)
sa_v2 = SelfAttention_V2(d_in, d_out)
print(sa_v2(inputs))