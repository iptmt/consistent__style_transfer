import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
n_head = 8
n_layer = 6

class MLM(nn.Module):
    def __init__(self, n_vocab, n_class):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.posit_embedding = nn.Embedding(100, d_model) # max sequence length -> 100
        self.style_embedding = nn.Embedding(n_class, d_model)

        self.lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), num_layers=n_layer
        )

        self.fwd = nn.Linear(d_model, n_vocab)
    
    # tensor: (B, L, *), label: (B,)
    def embedding(self, tensor, label):
        if len(tensor.shape) == 2:
            E_t = self.token_embedding(tensor)
        elif len(tensor.shape) == 3:
            E_t = tensor.matmul(self.token_embedding.weight)
        else:
            raise Exception
        p_idx = torch.arange(tensor.size(1), device=tensor.device).long().unsqueeze(0).expand(tensor.size(0), -1)
        E_p = self.posit_embedding(p_idx)
        E_s = self.style_embedding(label).unsqueeze(1)
        return E_t + E_p + E_s

    def forward(self, inputs, label, res_type="none", tau=1.0): # res_type: "none", "softmax", "gumbel"
        x = self.embedding(inputs, label)

        x = self.lm(x.transpose(0, 1)).transpose(0, 1)

        logits = self.fwd(x)

        if res_type == "none":
            return logits
        elif res_type == "softmax":
            return F.softmax(logits / tau, dim=-1)
        else:
            return F.gumbel_softmax(logits, tau=tau, hard=False)