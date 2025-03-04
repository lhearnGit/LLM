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

class SelfAttention_V1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        ##Initialize Matricies with randomized values, based on input & output dimension sizes
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key ## using embedded input values x perform matrix multiplication on initialized values
        queries = x @ self.W_query
        values = x @ self.W_value

        #Calculate the attention scores for the input x,
        attn_scores = queries @ keys.T # Omega Values
        #perform softmax calculation to reduce down to sums of 1
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
    

torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in, d_out)
print(sa_v1(inputs))