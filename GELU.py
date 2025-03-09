import torch.nn as nn
import torch
# import matplotlib.pyplot as plt

# GELU - Gaussian Error Linear Unit

class GELU (nn.Module):
    def __init__(self):
        super().__init__( )
    def forward(self,x):
        return .5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi )) *  
            ( x + .044715 * torch.pow(x,3))
            ))
    

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