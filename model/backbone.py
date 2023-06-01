import torch
import torch.nn as nn
from model.modules import MLP, SoftAttn, PE
from model.encoder import Encoder


class monitor(nn.Module):
    def __init__(self, hid_img_dim, cell_dim, attn_dim):
        super(monitor, self).__init__()
        self.w_hid_img = nn.Linear(hid_img_dim, cell_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.w_p = nn.Linear(attn_dim + cell_dim, 1)

    def forward(self, h_img, c, a):
        h = self.w_hid_img(h_img)
        h = self.sigmoid(h * self.tanh(c))
        h = torch.cat([a, h], dim=-1)
        return self.tanh(self.w_p(h))


class MonitorNet(nn.Module):
    def __init__(self, img_fc_l=(128, 1024), img_dropout=0.5, img_input=2176,
                 rnn_hidden=512, rnn_dropout=0.5, seq_len=80, max_nav=15, num_img=32, **enc_args):
        super(MonitorNet, self).__init__()
        self.max_neighbor = max_nav
        self.num_predefined_act = 1  # stop
        self.num_img = num_img
        self.img_dim = img_input
        self.h_dim = rnn_hidden

        img_fc = [img_input]
        img_fc.extend(img_fc_l)
        self.vis_mlp = MLP(img_fc, img_dropout)
        self.vis_attn = SoftAttn(img_fc[-1], rnn_hidden)

        self.encoder = Encoder(**enc_args)
        self.pe = PE(rnn_hidden, seq_len)
        self.text_attn = SoftAttn(rnn_hidden, rnn_hidden)

        self.h_lstm = nn.LSTMCell(img_fc[-1] + img_input + rnn_hidden, rnn_hidden)
        self.act_attn = SoftAttn(img_fc[-1], rnn_hidden*2)
        self.monitor = monitor(img_input + rnn_hidden, rnn_hidden, seq_len)

    def init_hidden_feat(self, bs):
        return torch.zeros(bs, self.h_dim).to("cuda:0"), torch.zeros(bs, self.h_dim).to("cuda:0")

    def _mask(self, bs, num_neighbor):
        mask = torch.zeros((bs, self.max_neighbor + self.num_predefined_act)).to(torch.float32)
        for i, n in enumerate(num_neighbor):
            mask[i, :n+self.num_predefined_act] = 1.
        return mask.to("cuda:0")

    # def pre_text_encode(self, seq, seq_len):
    #     bs = seq.shape[0]
    #     h, c = self.init_hidden_feat(bs)
    #     seq = self.encoder(seq, seq_len)
    #     tmp = self.pe(seq)
    #     a_seq = self.text_attn(tmp, h)
    #

    def forward(self, img, last_act_img, num_neighbor, seq, seq_len, h, c):
        """
        img: [b, k, f]
        last_act_img: [b,f]
        """
        bs = img.shape[0]
        mask = self._mask(bs, num_neighbor)[..., None]  # 3 dim
        pre_v = self.vis_mlp(last_act_img)

        # text attn
        seq = self.encoder(seq, seq_len)
        tmp = self.pe(seq)
        a_seq = self.text_attn(tmp, h)
        xt = torch.sum(a_seq * seq, dim=-2)

        # visual attn
        vis = self.vis_mlp(img) * mask
        a_vis = self.vis_attn(vis, h, mask)
        vt = torch.sum(a_vis * img, dim=-2)

        # action attn
        ht, ct = self.h_lstm(torch.cat([xt, vt, pre_v], dim=-1), (h, c))
        p = self.act_attn(vis, torch.cat([ht, xt], dim=-1)).squeeze()

        # monitor
        p_pm = self.monitor(torch.cat([h, vt], dim=-1), ct, a_seq.squeeze())
        return ht, ct, p, p_pm, mask.squeeze()



