import copy
import random
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm

from torch.nn.modules import MultiheadAttention

import torch.nn.functional as F

d_embed = 128
d_enc = 256
d_dec = 512
p_drop = 0.1

class DenoiseLSTM(nn.Module):
    def __init__(self, n_vocab, n_class, max_len):
        super().__init__()
        self.start_embedding = nn.Embedding(1, d_embed)
        self.token_embedding = nn.Embedding(n_vocab, d_embed)

        self.enc_style_embedding = nn.Embedding(n_class, 2 * d_enc)
        self.style_embedding = nn.Embedding(n_class, d_dec)

        self.encoder = nn.LSTM(
            input_size=d_embed, hidden_size=d_enc, num_layers=1,
            batch_first=True, bidirectional=True
        )

        self.decoder = nn.LSTM(
            input_size=d_embed, hidden_size=d_dec, num_layers=1,
            batch_first=True, bidirectional=False
        )

        self.transfer = nn.Linear(2 * d_enc, d_dec, bias=False)

        self.fn_1 = nn.Linear(2 * d_enc + d_dec, d_dec)
        self.fn_2 = nn.Linear(d_dec, n_vocab, bias=False)

        self.dropout = nn.Dropout(p_drop)
        self.relu = nn.LeakyReLU(0.1)

        self.max_len = max_len
    
    # q: b * 1 * H; k=v: b * L * H
    def dot_attn(self, q, k, v):
        a = q.bmm(k.transpose(1, 2)) # b * 1 * L
        a_norm = F.softmax(a / (k.size(-1) ** 0.5), dim=-1)
        c = a_norm.bmm(v) # b * 1 * H
        return c
    
    def forward(self, inp, label_i, x, label, res_type="none", tau=1.0):
        # encode
        h_0 = self.enc_style_embedding(label_i).reshape(-1, 2, d_enc).transpose(0, 1).contiguous()
        np = self.dropout(self.token_embedding(inp))
        memory, (_, c_end) = self.encoder(inp, (h_0, h_0.new_zeros(h_0.shape)))

        # decode
        max_len = self.max_len if x is None else x.size(1)

        x_t = self.start_embedding(inp.new_full((inp.size(0), 1), 0).long()) # B * 1 * d_emb
        c_t = self.relu(self.transfer(c_end.transpose(0, 1).reshape(1, inp.size(0), -1))) # 1 * B * d_dec
        h_t = self.style_embedding(label).unsqueeze(0) # 1 * B * d_dec

        logits = []
        for step in range(max_len):
            # update GRU
            o_t, (h_t, c_t) = self.decoder(x_t, (h_t, c_t))
            # update a_t
            a_t = self.dot_attn(o_t, memory, memory)

            i_ffn = torch.cat([o_t, a_t], dim=-1)
            o_f1 = self.fn_1(self.dropout(i_ffn))
            logits_t = self.fn_2(self.relu(o_f1))

            if res_type == "softmax":
                logits_t = F.softmax(logits_t / tau, dim=-1)
            elif res_type == "gumbel":
                logits_t = F.gumbel_softmax(logits_t, tau=tau, hard=False)
            logits.append(logits_t)

            if x is None or random.random() < 1/2:
                x_t = logits_t.argmax(-1)
            else:
                x_t = x[:, step].unsqueeze(1)
            x_t = self.dropout(self.token_embedding(x_t))
        return torch.cat(logits, dim=1)


if __name__ == "__main__":
    model = DenoiseLSTM(10000, 2, 16)
    nx = torch.randint(0, 9999, (64, 15))
    x = torch.randint(0, 9999, (64, 20))
    label = torch.randint(0, 1, (64,))
    output_0 = model(nx, 1-label, x, label, res_type="gumbel", tau=0.1)
    output_1 = model(nx, 1-label, None, label, res_type="softmax", tau=0.01)
    print(output_0.shape)
    print(output_1.shape)