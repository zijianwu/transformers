import torch
from torch import nn
import torch.functional as F

class SelfAttention(nn.module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        assert k % heads == 0 # embedding dim should be divisible by number of heads

        self.k = k
        self.heads = heads

        self.W_k = nn.Linear(k, k, bias=False)
        self.W_q = nn.Linear(k, k, bias=False)
        self.W_v = nn.Linear(k, k, bias=False)

        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        
        queries = self.W_q(x) # (b, t, k)
        keys = self.W_k(x) # (b, t, k)
        values = self.W_v(x) # (b, t, k)

        s = k // h

        queries = queries.reshape(b, t, h, s) 
        keys = keys.reshape(b, t, h, s) 
        values = values.reshape(b, t, h, s)

        keys = keys.transpose(1, 2).reshape(b * h, t, s)
        queries = queries.transpose(1, 2).reshape(b * h, t, s)
        values = values.transpose(1, 2).reshape(b * h, t, s)

        dot = torch.bmm(queries, keys.transpose(1, 2)) # (b*h, t, t)
        dot = dot / (k ** (1/2))
        dot = F.softmax(dot, dim=2) # (b290222h, t, t) row normalization

        out = torch.bmm(dot, values) # (b*h, t, s)
        out = out.reshape(b, h, t, s)

        out = out.transpose(1, 2).reshape(b, t, s * h)

        return self.unifyheads(out) # (b, t, k)


class AttentionBlock(nn.module):
    def __init__(self, k, heads=4):
        self.selfattention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.ff = nn.Sequential(nn.Linear(k, 4 * k),
                                nn.ReLU(),
                                nn.Linear(4 * k, k))
        self.norm2 = nn.LayerNorm(k) 

    def forward(self, x):
        attended = self.selfattention(x) 
        x = self.norm1(attended + x) 
        fedforward = self.ff(x) 
        x = self.norm2(fedforward + x) 
        return x # (b, t, k)


class Transformer(nn.module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        tblocks = []
        for _ in range(depth):
            tblocks.append(AttentionBlock(k, heads))
        self.transformer_blocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(k, num_classes) # maps output sequence to class logits

    def forward(self, x):
        """
        
        Arguments:
            x {torch.Tensor} -- (b, t) Tensor of integer values representing
            words in predefined dictionary

        Returns:
            torch.Tensor -- (b, c) tensor of log probabilities for each class
        """

        tokens = self.token_emb(x) # (b, t, k)
        b, t, k = tokens.size()

        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions

        x = self.transformer_blocks(x) # (b, t, k)

        x = x.mean(dim=1) # (b, k)

        x = self.toprobs(x) # (b, c)

        return F.log_softmax(x, dim=1)