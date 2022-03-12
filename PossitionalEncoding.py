"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


from torch import nn, Tensor
import torch
import math



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        pre and others tensors here are defined with required_grad = False since they are
        using inplace memory so they cant save the grad. Also we dont need them as they are
        helper tensors for forward inputs. Thus specify allow_unused=True for gradients.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        return x