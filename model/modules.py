import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim, dp_ratio):
        super(MLP, self).__init__()
        modules = []
        for i in range(len(dim)-2):
            modules.append(nn.Linear(dim[i], dim[i+1]))
            modules.append(nn.Dropout(dp_ratio))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(dim[-2], dim[-1]))
        self.f = nn.Sequential(*modules)

    def forward(self, x):
        return self.f(x)


class SoftAttn(nn.Module):
    def __init__(self, mlp_dim, hidden_size):
        super(SoftAttn, self).__init__()
        self.hidden_W = nn.Linear(hidden_size, mlp_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, h, mask=None, return_attn=True):
        tmp = torch.sum(self.hidden_W(h).unsqueeze(1) * x, dim=-1).unsqueeze(-1)
        if mask is not None:
            tmp.data.masked_fill_((mask.expand_as(tmp) == 0).data, float("-inf"))
        if return_attn:
            return self.softmax(tmp.squeeze()).unsqueeze(-1)
        return torch.sum(self.softmax(tmp) * x, dim=-2)


class PE(nn.Module):
    # PE(pos, 2i) = sin(pos/10000^(2i/emb_dim))
    # PE(pos, 2i+1) = cos(pos/10000^(2i/emb_dim))
    def __init__(self, dim, seq_len=80):
        super(PE, self).__init__()
        pe = torch.zeros(1, seq_len, dim)
        pos = torch.arange(0, seq_len).float().unsqueeze(1)
        k = np.power(10000., np.arange(0, dim, 2)[None, :] / np.float32(seq_len))
        pe[0, :, 0::2] = torch.sin(pos / k)
        pe[0, :, 1::2] = torch.cos(pos / k[:, :dim//2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.pe[:, :x.shape[1]]
        return x


if __name__ == "__main__":
    # bs = 2
    # num_img = 4
    # dim_img = 10
    # dim_hid = 3
    # s = SoftAttn([dim_img, 8, 6], dim_hid)
    # hid = torch.rand((bs, dim_hid))
    # img = torch.rand((bs, num_img, dim_img))
    #
    # print(s(img, hid).shape)
    pe = PE(3, 30)
    print(pe(torch.arange(24).reshape((2, 4, 3)).to(torch.float32)))
    print(pe(torch.zeros(2, 4, 3).to(torch.float32)))
