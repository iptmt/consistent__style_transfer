import copy
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, Dropout

from torch.nn.modules import MultiheadAttention

import torch.nn.functional as F


class DenoiseTransformer(nn.Module):
    def __init__(self, n_vocab, n_class, seq_max_len, d_model=512, n_head=8,
                       n_enc_layer=4, n_dec_layer=6, p_dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.posit_embedding = nn.Embedding(100, d_model)
        self.style_embedding = nn.Embedding(n_class, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), 
            num_layers=n_enc_layer
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head), 
            num_layers=n_dec_layer
        )

        self.proj_to_vocab = nn.Linear(d_model, n_vocab)

        self.n_vocab = n_vocab
        self.max_len = seq_max_len

        self.dropout = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.token_embedding.weight)
    
    def encoder_embed(self, x):
        E_t = self.token_embedding(x)
        E_p = self.posit_embedding(torch.arange(x.size(1), device=x.device).long()).unsqueeze(0)
        return E_t + E_p
    
    def decoder_embed(self, label, max_len):
        E_s = self.style_embedding(label).unsqueeze(1)
        E_p = self.posit_embedding(torch.arange(max_len, device=label.device).long()).unsqueeze(0)
        return E_s + E_p
    
    def forward(self, x, label, max_len=None, gumbel=False, tau=1.0):
        max_len = max_len if max_len is not None else self.max_len

        x = self.dropout(self.encoder_embed(x))
        y = self.decoder_embed(label, max_len)

        memory = self.encoder(x.transpose(0, 1))

        output = self.decoder(y.transpose(0, 1), memory)

        logits = self.proj_to_vocab(self.dropout(output.transpose(0, 1)))

        if gumbel:
            p_sample = F.gumbel_softmax(logits, tau=tau, hard=False)
            return p_sample
        else:
            return logits


class DNTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DNTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # generate
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)
        # denoise
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




if __name__ == "__main__":
    model = DenoiseTransformer(10000, 2, 16)
    x = torch.randint(0, 10000, (8, 14))
    l = torch.randint(0, 1, (8,))
    o = model(x, l)
    print(o.shape)