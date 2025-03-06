import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1,2)
        # drop out masking, to prevent model overfit, disabled after training
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax( attention_scores / keys.shape[-1]**.5, dim=-1)

        attention_weights = self.dropout(attention_weights) 
        context_vector = attention_weights @ values
        return context_vector



inputs = torch.tensor([
    [ .43 , .15 , .89], #Your
    [ .55, .87 , .66],  #journey
    [ .57 , .85 , .64], #starts
    [ .22 , .58 , .33], #with
    [ .77 , .25 , .10], #one
    [ .05 , .80 , .55], #step
])
torch.manual_seed(123)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1] # embedding size d=3
d_out = 2 # sets output dimension
print(batch.shape)


ca = CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0)
context_vector = ca(batch)
print("Context_Vector.shape",context_vector.shape)