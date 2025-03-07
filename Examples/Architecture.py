import torch.nn as nn
import torch
import tiktoken


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out,
                   context_length, dropout, num_heads, qkv_bias=False):
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

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
             nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
    def forward(self,x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape: 
                x = x + layer_output
            else:
                    x = layer_output
        return x
        

inputs = torch.tensor([
    [ .43 , .15 , .89], #Your
    [ .55, .87 , .66],  #journey
    [ .57 , .85 , .64], #starts
    [ .22 , .58 , .33], #with
    [ .77 , .25 , .10], #one
    [ .05 , .80 , .55], #step
])


class GELU (nn.Module):
    def __init__(self):
        super().__init__( )
    def forward(self,x):
        return .5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi )) *  
            ( x + .044715 * torch.pow(x,3))
            ))
    

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            #expand and then contract back after the  GELU function
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self,x):
         return self.layers(x)
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self,x):
            return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #small constant epsilon to prevent div by zero during normalization
        # scale and shift are two trainable parameters that are adjusted during training automatically, if determined it would improve performance
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
            
            mean = x.mean(dim=-1, keepdim=True) 
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            
            return self.scale * norm_x + self.shift


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # Place Holders
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        # Place Holder
        self.final_norm = LayerNorm(cfg["emb_dim"])


        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_leng = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_leng, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch,dim=0)

print(batch)
GPT_CONFIG_124M = {
    "vocab_size":50257, #Vocab Size, number of BPE Tokens
    "context_length":1024, #minimum context block size for gpt2
    "emb_dim":768, #minimum embed dims for gpt2
    "n_heads":12, 
    "n_layers":12,
    "drop_rate":.1,
    "qkv_bias":False
}

##model = DummyGPTModel(GPT_CONFIG_124M)

##logits = model(batch)

#print("output shape", logits.shape)
#print(logits)
# ----------------- Layer Normalization Steps 
ex_batch = torch.randn(2,5)
# layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
#out = layer(ex_batch)
# print("out ex ", out)

# keep dim true, output tensor retains the same dimensions as input tensor
# dim = 0 get a mean across rows one per column
# dim = -1 or 1, get a mean across the columns, one per row
# 1 2  dim=1 = r1Mean = mean of 1,2  r2Mean = mean of 3,4 
# 3 4  dim = 0 c1 mean = 1,3 c2 = 2,4

#mean = out.mean(dim=-1, keepdim=True)
#var = out.var(dim=-1, keepdim=True)

#out_norm = (out - mean)/ torch.sqrt(var)
#mean = out_norm.mean(dim=-1, keepdim=True)
#var = out_norm.var(dim=-1, keepdim=True)

#torch.set_printoptions(sci_mode=False)
#print("Normalized  \n ", out_norm )
#print("Mean \n ", mean )
#print("Variance \n ", var)


# Layer Normalization Class Test ---------

ln = LayerNorm(emb_dim=5)
out_ln = ln(ex_batch)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased = False, keepdim=True)

torch.set_printoptions(sci_mode=False)
print("Normalized  \n ", out_ln )
print("Mean \n ", mean )
print("Variance \n ", var)



# GELU & RELU EXAMPLE WILL OPEN GRAPHS

# gelu, relu = GELU(), nn.ReLU()
# # Some sample data
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)

# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)

# plt.tight_layout()
# plt.show()



# feed forward network example
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2,3,768)
out = ffn(x) # will expand the inputs by a factor of 4 - > 3072 perform an operation, then contract back down
print(out.shape) # 2,3, 768 