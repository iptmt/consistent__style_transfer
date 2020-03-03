import math
import torch
import torch.nn as nn

from transformers import PreTrainedModel, AlbertConfig, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP

class MultiLinear(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(MultiLinear, self).__init__()

        self.linear_list = nn.ModuleList(
            [nn.Linear(dim, output_dim) for dim in input_dims]
        )

        self.linear = nn.Linear(output_dim * len(input_dims), output_dim)
        self.relu = nn.ReLU()

    def forward(self, tensors):
        pairs = zip(tensors, self.linear_list)
        
        outputs = [module(tensor) for tensor, module in pairs]

        outputs = torch.cat(outputs, dim=-1)

        outputs = self.linear(self.relu(outputs))

        return outputs



class Attention(nn.Module):
    def __init__(self, dim_q, dim_k):
        super(Attention, self).__init__()

        self.U = nn.Linear(dim_k, dim_q, bias=False)
        self.W = nn.Linear(dim_q, dim_q, bias=False)

        self.v = nn.Parameter(torch.full((dim_q,), dim_q ** -0.5))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
    
    """q: N * 1 * D_q; k: N * T * D_k; v: N * T * D_v
    """ 
    def forward(self, q, k, v, weights_only=False):
        weights = self.tanh(self.W(q) + self.U(k)).matmul(self.v) # N * T
        if weights_only:
            return weights

        norm_weights = self.softmax(weights)

        return norm_weights.unsqueeze(1).bmm(v)