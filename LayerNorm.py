import torch.nn as nn
import torch
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

