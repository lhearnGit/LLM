import torch
import os
import urllib
import tiktoken


import matplotlib.pyplot as plt
# Locals

from GPTModel import GPTModel
from gpt_download import download_and_load_gpt2
from Generate import generate
from AssignAndLoad import load_weights_into_gpt
from TextToIdFns import text_to_token_ids, token_ids_to_text

def main(gpt_config, input_prompt, model_size):

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings,params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":

    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-xl (1558M)"
    INPUT_PROMPT = "Every effort moves you"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate, Drop-out rate zero'd as the model is not training
        "qkv_bias": True         # Query-key-value bias, OpenAI used Bias for training on GPT2, current models do not typically include
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size)