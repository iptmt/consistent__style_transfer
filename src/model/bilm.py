import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
n_head = 8
n_layer = 4

class BiLM(nn.Module):
    def __init__(self, n_vocab, n_class):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.posit_embedding = nn.Embedding(100, d_model) # max sequence length -> 100
        self.style_embedding = nn.Embedding(n_class, d_model)

        self.lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), num_layers=n_layer
        )

        self.hidden2logits = nn.Linear(d_model, n_vocab)
    
    # tensor: (B, L, *); style: (B)
    def embedding(self, tensor, style):
        if len(tensor.shape) == 2:
            E_t = self.token_embedding(tensor)
        elif len(tensor.shape) == 3:
            E_t = tensor.matmul(self.token_embedding.weight)
        else:
            raise Exception
        E_s = self.style_embedding(style).unsqueeze(1)
        p_idx = torch.arange(tensor.size(1), device=tensor.device).long().unsqueeze(0).expand(tensor.size(0), -1)
        E_p = self.posit_embedding(p_idx)
        return E_t + E_s + E_p

    def fwd_mask(self, tensor):
        length = tensor.size(1)
        mask = torch.full((length, length), float("-inf"), device=tensor.device)
        return mask.triu(diagonal=1)

    def bwd_mask(self, tensor):
        return self.fwd_mask(tensor).T
        
    def fwd_pad(self, tensor):
        return F.pad(tensor, [0, 0, 1, 0])
    
    def bwd_pad(self, tensor):
        return F.pad(tensor, [0, 0, 0, 1])
    
    def forward(self, inputs, labels):
        x = self.embedding(inputs, labels)
        x_f, x_b = x[:, :-1, :], x[:, 1:, :]

        # forward lm
        x_f = self.lm(x_f.transpose(0, 1), mask=self.fwd_mask(x_f)).transpose(0, 1)

        # backward lm
        x_b = self.lm(x_b.transpose(0, 1), mask=self.bwd_mask(x_b)).transpose(0, 1)

        x = self.fwd_pad(x_f) + self.bwd_pad(x_b)

        return self.hidden2logits(x)