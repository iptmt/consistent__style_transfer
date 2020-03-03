import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.layers import MultiLinear, Attention



class RNNSearch(nn.Module):
    def __init__(self, n_vocab, d_embed, d_enc_hidden, d_dec_hidden, n_enc_layer, n_dec_layer,
                       n_class, p_drop, max_len):
        super(RNNSearch, self).__init__()

        self.d_embed = d_embed

        self.d_enc_hidden = d_enc_hidden
        self.d_dec_hidden = d_dec_hidden

        self.n_enc_layer = n_enc_layer
        self.n_dec_layer = n_dec_layer

        self.token_embedding = nn.Embedding(n_vocab, d_embed)
        self.style_embedding = nn.Embedding(n_class, n_dec_layer * d_dec_hidden)
        self.start_embedding = nn.Embedding(1, d_embed)

        self.encoder = nn.GRU(
            input_size=d_embed, hidden_size=d_enc_hidden, num_layers=n_enc_layer,
            batch_first=True, bidirectional=True,
        )

        self.decoder = nn.GRU(
            input_size=d_dec_hidden, hidden_size=self.d_dec_hidden,
            num_layers=n_dec_layer, batch_first=True, bidirectional=False,
        )

        self.transfer_output = nn.Linear(2 * n_enc_layer * d_enc_hidden, d_dec_hidden)
        self.context_attn = Attention(self.d_dec_hidden, 2 * d_enc_hidden)

        #1.token info; 2.c_t
        self.fuse = MultiLinear([d_embed, 2 * d_enc_hidden], self.d_dec_hidden)
        
        self.proj_vocab = nn.Linear(self.d_dec_hidden, n_vocab)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=p_drop)
        self.relu = nn.ReLU()

        self.max_len = max_len + 1
    
    def forward(self, input_ids, labels, teacher_ids, max_len, gumbel=False, tau=1.0):
        # encode
        if len(input_ids.shape) == 2:
            enc = self.dropout(self.token_embedding(input_ids))
        else:
            enc = self.dropout(input_ids.matmul(self.token_embedding))

        memory, h_0 = self.encoder(enc, None)

        bsz, _, _ = memory.shape

        # decode
        embed_t = self.start_embedding(input_ids.new_zeros((bsz, 1), dtype=torch.long))
        output_t = self.transfer_output(self.relu(h_0.transpose(0, 1).reshape(bsz, -1))).unsqueeze(1)
        h_t = self.style_embedding(labels).reshape(bsz, self.n_dec_layer, -1).transpose(0, 1).contiguous()

        max_seq_len = max_len if max_len is not None else self.max_len
        
        output_logits, gumbel_sample = [], []
        for step in range(max_seq_len):
            # context
            c_t = self.context_attn(output_t, memory, memory)

            fused_input_t = self.fuse([self.dropout(embed_t), self.dropout(c_t)])

            output_t, h_t = self.decoder(fused_input_t, h_t)

            vocab_logits = self.proj_vocab(output_t) # N * 1 * V

            output_logits.append(vocab_logits)

            if teacher_ids is None or random.random() < 2/3:
                if gumbel:
                    p_t = F.gumbel_softmax(vocab_logits, tau=tau, hard=False)
                    gumbel_sample.append(p_t)
                    x_t = p_t.argmax(-1)
                else:
                    x_t = vocab_logits.argmax(-1)
            else:
                x_t = teacher_ids[:, min(step, max_seq_len - 1)].unsqueeze(1)
            embed_t = self.dropout(self.token_embedding(x_t))

        return torch.cat(output_logits, dim=1), torch.cat(gumbel_sample, dim=1) if gumbel_sample else None