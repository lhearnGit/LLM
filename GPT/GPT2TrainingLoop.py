import torch
import tiktoken
from GPT2Model import GPT2Model
from GPT2Parts.DataLoader import create_data_loader
from GPT2Parts.Train_Model_Simple import train_model_simple
from GPT2Parts.PlotLosses import plot_losses
import time
start_time = time.time()


file_path = "LLM/Data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


GPT_CONFIG_124M = {
    "vocab_size":50257, #Vocab Size, number of BPE Tokens
    "context_length":256, #minimum context block size for gpt2
    "emb_dim":768, #minimum embed dims for gpt2
    "n_heads":12, 
    "n_layers":12,
    "drop_rate":.1,
    "qkv_bias":False
}

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2Model(GPT_CONFIG_124M)
model.eval()


# Validate File Loads Properly
# print(text_data[:99])

# Tokenize File and Output 
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print('Character count : ' , total_characters)
# print('Token Count : ' , total_tokens)



train_ratio = .90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_data_loader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_data_loader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
model = GPT2Model(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)