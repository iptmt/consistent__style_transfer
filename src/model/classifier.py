import torch
import torch.nn as nn

import torch.nn.functional as F

d_embed = 128
p_drop = 0.5
kernels=[3, 4, 5]
kernel_number=[128, 128, 128]


class TextCNN(nn.Module):
    def __init__(self, n_vocab, n_class):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, d_embed)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, number, (size, d_embed), padding=(size-1, 0)) for (size, number) in zip(kernels, kernel_number)]
        )
        self.dropout=nn.Dropout(p_drop)
        self.out = nn.Linear(sum(kernel_number), n_class)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.embedding(x)
        elif len(x.shape) == 3:
            x = x.matmul(self.embedding.weight)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        if len(x.shape) == 2:
            x = self.out(self.dropout(x))
        elif len(x.shape) == 3:
            x = self.out(x)

        return x


if __name__ == "__main__":
    model = TextCNN(n_vocab=10000)
    x = torch.randint(9999, (64, 15))
    logits = model(x)
    print(logits)
