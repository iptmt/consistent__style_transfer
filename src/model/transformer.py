import torch
import torch.nn as nn

import torch.nn.functional as F


class DenoiseTransformer(nn.Module):
    def __init__(self, n_vocab, n_class, seq_max_len, d_model=512, n_head=8, 
                       n_enc_layer=6, n_dec_layer=6, p_dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Linear(n_vocab, d_model, bias=False)
        self.posit_embedding = nn.Embedding(100, d_model)
        self.style_embedding = nn.Embedding(n_class, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), num_layers=n_enc_layer
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head), num_layers=n_dec_layer
        )

        self.n_vocab = n_vocab
        self.max_len = seq_max_len

        self.dropout = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.token_embedding.weight)
    
    def encoder_embed(self, x):
        x_oh = F.one_hot(x, self.n_vocab).float()
        E_t = self.token_embedding(x_oh)
        E_p = self.posit_embedding(torch.arange(x.size(1), device=x.device).long()).unsqueeze(0)
        return E_t + E_p
    
    def decoder_embed(self, label, max_len):
        E_s = self.style_embedding(label).unsqueeze(1)
        E_p = self.posit_embedding(torch.arange(max_len, device=label.device).long()).unsqueeze(0)
        return E_s + E_p
    
    def proj_to_vocab(self, h):
        return h.matmul(self.token_embedding.weight)
    
    def forward(self, x, label, max_len=None, gumbel=False, tau=1.0):
        max_len = max_len if max_len is not None else self.max_len

        x = self.dropout(self.encoder_embed(x))
        lb = self.decoder_embed(label, max_len)

        memory = self.encoder(x.transpose(0, 1))
 
        output = self.decoder(lb.transpose(0, 1), memory)

        logits = self.proj_to_vocab(self.dropout(output.transpose(0, 1)))

        if gumbel:
            p_sample = F.gumbel_softmax(logits, tau=tau, hard=False)
            return p_sample
        else:
            return logits


if __name__ == "__main__":
    model = DenoiseTransformer(10000, 2, 16)
    x = torch.randint(0, 10000, (8, 14))
    l = torch.randint(0, 1, (8,))
    o = model(x, l)
    print(o.shape)