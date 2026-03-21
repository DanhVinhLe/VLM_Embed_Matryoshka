import torch

# Create a batch of 3x3 matrices with 2 batch dimensions (e.g., 2, 4, 3, 3)
A = torch.randn(2, 4, 3, 3) 

# Perform batch-wise SVD
U, S, Vh = torch.linalg.svd(A)

# U, S, and Vh will have the same batch dimensions (2, 4, ...)
print(f"Shape of input A: {A.shape}")
print(f"Shape of U: {U.shape}")     # (2, 4, 3, 3)
print(f"Shape of S: {S.shape}")     # (2, 4, 3) - singular values are 1D
print(f"Shape of Vh: {Vh.shape}")   # (2, 4, 3, 3)
