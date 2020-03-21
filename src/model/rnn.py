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

        self.multi_attn = MultiheadAttention(d_dec, 8)

        self.project = nn.Linear(d_dec, n_vocab)

        self.norm = LayerNorm(d_dec)
        self.dropout = nn.Dropout(p_drop)
        self.tanh = nn.Tanh()

        self.max_len = max_len
    
    def forward(self, nx, x, label, res_type="none", tau=1.0):
        # encode
        nx = self.dropout(self.token_embedding(nx))
        memory, (h_end, _) = self.encoder(nx)
        memory = memory.transpose(0, 1) # transpose for multi-head attn

        # decode
        max_len = self.max_len if x is None else x.size(1)

        x_t = self.start_embedding(nx.new_full((nx.size(0), 1), 0).long()) # B * 1 * d_emb
        h_t = self.tanh(self.transfer(h_end.transpose(0, 1).reshape(1, nx.size(0), -1))) # 1 * B * d_dec
        c_t = self.style_embedding(label).unsqueeze(0) # 1 * B * d_dec

        logits = []
        for step in range(max_len):
            # update GRU
            _, (h_t_, c_t) = self.decoder(x_t, (h_t, c_t))
            # update a_t
            a_t, _ = self.multi_attn(h_t_, memory, memory)
            # update h_t
            h_t = self.norm(h_t_ + a_t)
            # update logits
            logits_t = self.project(h_t.transpose(0, 1))

            if res_type == "softmax":
                logits_t = F.softmax(logits_t / tau, dim=-1)
            elif res_type == "gumbel":
                logits_t = F.gumbel_softmax(logits_t, tau=tau, hard=True)
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
    output_0 = model(nx, x, label, res_type="gumbel", tau=0.1)
    output_1 = model(nx, None, label, res_type="softmax", tau=0.01)
    print(output_0.shape)
    print(output_1.shape)