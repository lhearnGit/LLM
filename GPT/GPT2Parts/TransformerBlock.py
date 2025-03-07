import torch.nn as nn
from GPT2Parts.FeedForward import FeedForward
from GPT2Parts.LayerNorm import LayerNorm
from GPT2Parts.MultiHeadedAttention import MultiHeadedAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadedAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


# torch.manual_seed(123)

# x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)