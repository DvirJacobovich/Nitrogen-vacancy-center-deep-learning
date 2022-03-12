"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


import torch
import torch.nn as nn
import TransfoermerModel
import flags as _

# vocab = None  # TODO what is vocab in our case
# ntokens = 200  # len(vocab)  # size of vocabulary. In our case we have 200 ("words") output size.
# emsize = 200  # embedding dimension
# # d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
# d_hid = 1  # 200 dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 20  # number of heads in nn.MultiheadAttention
# dropout = 0.2  # dropout probability


class NV_lorentzian_transformer(nn.Module):
    def __init__(self):
        super(NV_lorentzian_transformer, self).__init__()
        self._ntokens = _.FLAGS['_ntokens']
        self._emsize = _.FLAGS['_emsize']
        self._d_hid = _.FLAGS['_d_hid']
        self._nlayers = _.FLAGS['_nlayers']
        self._nhead = _.FLAGS['_nhead']  
        self._dropout = _.FLAGS['_dropout']

        self.Transformer_model = TransfoermerModel.TransformerModel(
            self._ntokens, self._emsize, self._nhead, self._d_hid, self._nlayers, self._dropout).to(_.FLAGS['device'])
        # self.mask = torch.triu(torch.ones(200, 200) * float('-inf'), diagonal=1)
        self.mask = torch.torch.ones(200, 200).to(_.FLAGS['device'])  # TODO this is unused
        self.mask.requires_grad = True
        self.fc = nn.Linear(200, 1)
        self.batch_nrm = nn.BatchNorm1d(200)
        self.batch_nrm2 = nn.BatchNorm1d(1)
 
    def forward(self, x):
        predict = self.Transformer_model(x, src_mask=self.mask)
        predict1 = self.batch_nrm(predict)  # TODO ?
        predict2 = self.fc(predict1)
        return self.batch_nrm2(torch.transpose(predict2, dim1=2, dim0=1))