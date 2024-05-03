import torch
import torch.nn.functional as F

B = 64
T = 32
K = 128

x = torch.randn(B, T, K)

raw_weights = torch.bmm(x, x.transpose(1, 2))  # (B, T, T)
weights = F.softmax(raw_weights, dim=2)  # (B, T, T)
y = torch.bmm(weights, x)  # (B, T, K)

W_q = torch.randn(K, K)
W_k = torch.randn(K, K)
W_v = torch.randn(K, K)

q = torch.matmul(W_q, x)  # (B, T, K)