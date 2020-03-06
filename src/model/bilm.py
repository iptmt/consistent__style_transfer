import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
n_head = 8
n_layer = 4

class BiLM(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.posit_embedding = nn.Embedding(100, d_model) # max sequence length -> 100

        self.lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head), num_layers=n_layer
        )

        self.fwd_1 = nn.Linear(2 * d_model, d_model)
        self.fwd_2 = nn.Linear(d_model, n_vocab)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
    
    # tensor: (B, L, *)
    def embedding(self, tensor):
        if len(tensor.shape) == 2:
            E_t = self.token_embedding(tensor)
        elif len(tensor.shape) == 3:
            E_t = tensor.matmul(self.token_embedding.weight)
        else:
            raise Exception
        p_idx = torch.arange(tensor.size(1), device=tensor.device).long().unsqueeze(0).expand(tensor.size(0), -1)
        E_p = self.posit_embedding(p_idx)
        return E_t + E_p

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
    
    def forward(self, inputs):
        x = self.embedding(inputs)
        x_f, x_b = x[:, :-1, :], x[:, 1:, :]

        # forward lm
        x_f = self.lm(x_f.transpose(0, 1), mask=self.fwd_mask(x_f)).transpose(0, 1)

        # backward lm
        x_b = self.lm(x_b.transpose(0, 1), mask=self.bwd_mask(x_b)).transpose(0, 1)

        x = torch.cat([self.fwd_pad(x_f), self.bwd_pad(x_b)], dim=-1)

        return self.fwd_2(self.relu(self.fwd_1(self.drop(x))))


if __name__ == "__main__":
    model = BiLM(10000)
    input_int = torch.randint(0, 9999, (64, 20))
    input_float = torch.randn((64, 21, 10000))
    o = model(input_int)
    print(o.shape)
    o = model(F.softmax(input_float))
    print(o.shape)
