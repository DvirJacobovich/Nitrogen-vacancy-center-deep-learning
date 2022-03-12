"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import PossitionalEncoding as pos
import math
import flags as _


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = pos.PositionalEncoding(d_model, dropout).to(_.FLAGS['device'])
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True).to(_.FLAGS['device'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(_.FLAGS['device'])
        self.encoder = nn.Embedding(ntoken, d_model).to(_.FLAGS['device'])  # try to give up on it..
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.fc2 = nn.Linear(1, 200)
        self.batch_nrm1 = nn.BatchNorm1d(200)
        self.batch_nrm2 = nn.BatchNorm1d(200)
        self.batch_nrm3 = nn.BatchNorm1d(200)
        self.batch_nrm4 = nn.BatchNorm1d(200)
        self.init_weights()

    def init_weights(self) -> None:  # TODO to all
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # fc = nn.Linear(src.size()[0], 200)
        # src_decoder = self.encoder(src) * math.sqrt(self.d_model) # TODO at the end of the file there is extension.

        tr = torch.transpose(src, dim0=2, dim1=1)
        fc_out = self.fc2(tr)
        # size: (batch_size, Channels, seq) = (batch_size, 200, 200)
        tr1 = self.batch_nrm1(fc_out)
        
        src0 = self.pos_encoder(tr1)
        src1 = self.batch_nrm2(src0)

        mask = torch.torch.ones(200, 200).to(_.FLAGS['device'])
        mask.requires_grad = True

        output1 = self.batch_nrm3(self.transformer_encoder(src1, mask))
        output3 = self.batch_nrm4(self.decoder(output1))
        return output3




def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# min_long = torch.min(src)
# max_long = torch.max(src)
# long_src = (src - min_long) / max_long
# long_src = long_src * 200
# long_src2 = long_src.long()
# self.encoder(long_src2) * math.sqrt(self.d_model)
