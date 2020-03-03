import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
n_head = 8
n_layer = 6


class Matcher(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.posit_embedding = nn.Embedding(100, d_model) # max sequence length -> 100

        self.matcher = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), num_layers=n_layer
        )

        self.hidden2logits = nn.Linear(d_model, 1)
    
    def embedding(self, tensor, seg_id):
        if len(tensor.shape) == 2:
            E_t = self.token_embedding(tensor)
        elif len(tensor.shape) == 3:
            E_t = tensor.matmul(self.token_embedding.weight)
        else:
            raise Exception
        p_idx = torch.arange(tensor.size(1), device=tensor.device).long().unsqueeze(0).expand(tensor.size(0), -1)
        E_p = self.posit_embedding(p_idx)
        E_s = self.segment_embedding(torch.full((tensor.shape[:-1]), seg_id, device=tensor.device).long()).unsqueeze(1)
        return E_t + E_p + E_s
    
    def forward(self, x1, x2):
        x = torch.cat([self.embedding(x1, 0), self.embedding(x2, 1)], dim=1)
        
        x = self.matcher(x.transpose(0, 1)).transpose(0, 1)

        x_pool, _ = x.max(dim=1)

        return self.hidden2logits(x_pool).squeeze(1)