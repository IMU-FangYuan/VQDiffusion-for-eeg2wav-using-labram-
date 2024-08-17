#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:43:09 2024

@author: fangyuan
"""

# -*- coding: utf-8 -*-
'''
from bert 
https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model/embedding
positionalEmbedding has maxlen
'''
import torch
import math
import torch.nn as nn



class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=8192):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


""" in vq diffusion embedsize should add 1 mask """
class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        vocab_size = vocab_size+1
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.vocab_size =  vocab_size
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        #print(self.token(sequence).shape, self.position(sequence).shape )
        x = self.token(sequence) + self.position(sequence) 
        return self.dropout(x)
    
class ChunkedEmbedding(nn.Module):

    def __init__(self, codebook_size: int, n_quantizer: int, dim: int) -> None:
        super().__init__()
        assert dim % n_quantizer == 0, f"ChunkedEmbedding output dim ({dim}) must be divisible by n_quant {n_quantizer}"
        self.embs = nn.ModuleList([nn.Embedding(codebook_size, dim//n_quantizer) for _ in range(n_quantizer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Embeds each codebook index in `x` (bs, seq_len, n_quantizer) to an embedding vector, concatenating results.
        Returns output of shape (bs, seq_len, dim)
        """
        y = torch.cat([self.embs[i](x[..., i]) for i in range(x.shape[-1])], dim=-1)
        return y



class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Assumes x of shape (bs, seq_len, dim) """
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


if __name__ == '__main__':
    # define model
    emblayer = BERTEmbedding(8124, 128)
    data = torch.randint(1,8044,[3,500])
    out = emblayer(data)
    print(out.shape)
    
    chunkemblayer = ChunkedEmbedding(8164, 8,512 )
    data = torch.randint(1,8100,[3,100,8])
    out = chunkemblayer(data)
    print("chunk emb",out.shape)
