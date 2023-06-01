import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=512, padding_idx=None,
                 dropout_ratio=0.5, bidirectional=False, num_layers=1):
        super(Encoder, self).__init__()
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.apply(self._init_param)

    def _init_param(self):
        pass

    def forward(self, x, length):
        x = self.emb(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        o, (h, c) = self.rnn(x_pack)
        o = pad_packed_sequence(o, batch_first=True)
        return o[0]



