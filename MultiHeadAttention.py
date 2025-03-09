import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # view Reshapes the original vector, without duplicating it in memory
        keys = keys.view(b,num_tokens, self.num_heads, self.head_dim)
        values = values.view(b,num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b,num_tokens, self.num_heads, self.head_dim)

        # transpose swaps axis 1, 2 
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
       # print("queries size", queries.size())
        values = values.transpose(1,2)

      
        # Queries @ Keys.transpose(2,3) = [2,2,6,1] @ [2,2,1,6] = [2,2,6,6]
        attention_scores = queries @ keys.transpose(2,3)   # Swap Axis - 2 and 3 - > shape [2,2,1,6] keys was transposed to [2,2,6,1 above]
        # print("Scores Size ", attention_scores.size())
        masked_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(masked_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2)

        context_vector = context_vector.contiguous().view(b,num_tokens,self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector
    

# inputs = torch.tensor([
#     [ .43 , .15 , .89], #Your
#     [ .55, .87 , .66],  #journey
#     [ .57 , .85 , .64], #starts
#     [ .22 , .58 , .33], #with
#     [ .77 , .25 , .10], #one
#     [ .05 , .80 , .55], #step
# ])


# torch.manual_seed(123)

# batch = torch.stack((inputs, inputs), dim=0)
# batch_size, context_length,d_in = batch.shape
# d_out = 2 # sets output dimension
# print(batch.shape)


# mha = MultiHeadedAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=2)
# context_vector = mha(batch)
# print("Context_Vector.shape",context_vector.shape)
# print("Context_Vector values \r\n",context_vector)