
import torch.nn as nn
import torch
from GPT2Parts.TransformerBlock import TransformerBlock
from GPT2Parts.LayerNorm import LayerNorm


GPT_CONFIG_124M = {
    "vocab_size":50257, #Vocab Size, number of BPE Tokens
    "context_length":1024, #minimum context block size for gpt2
    "emb_dim":768, #minimum embed dims for gpt2
    "n_heads":12, 
    "n_layers":12,
    "drop_rate":.1,
    "qkv_bias":False
}

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])


        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )


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
