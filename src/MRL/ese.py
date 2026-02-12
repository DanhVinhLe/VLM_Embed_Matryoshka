from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

class ESELoss:
    def __init__(self, 
                 temperature: float = 0.02, 
                 nested_dims: List[int] = (64, 128, 256, 512, 1024),
                 alpha: float = 1.0, 
                 beta: float = 1.0,
                 average_loss: bool = False):
        self.temperature = temperature
        self.nested_dims = nested_dims
        self.alpha = alpha
        self.beta = beta
        self.average_loss = average_loss
        
    @staticmethod
    def layer_weight(i : int) -> float:
        # w_i = 1.0 / (1.0 + math.log(i)), i is 1-index of layer, starting from 1
        return 1.0 /(1.0 + math.log(i))
    
    @staticmethod
    def pca_compress(embeddings: Tensor, k: int) -> Tensor:
        d = embeddings.size(-1)
        # A^d = softmax(X^d · X^d^T / sqrt(d))
        X = embeddings.unsqueeze(-1)
        inner_dep = torch.bmm(X, X.transpose(-1, -2)) / math.sqrt(d)
        A_d = F.softmax(inner_dep, dim=-1)
        U, S, Vt = torch.linalg.svd(A_d, full_matrices=False)
        
        A_k = U[:, : d, :k] * S[:, :k].unsqueeze(1) # Shape: (bs, d, k)
        
        X_k_pca = torch.bmm(A_k.transpose(-1, -2), X).squeeze(-1) # Shape: (bs, k)
        return X_k_pca
    
    @staticmethod
    def align_loss(x: Tensor, y: Tensor) -> Tensor:
        """align(x, y) = MSE(x, y) + KLDiv(x, y). y is detached soft label."""
        mse = F.mse_loss(x, y)
        log_p = F.log_softmax(x, dim=-1)
        q = F.softmax(y, dim=-1)
        kl = F.kl_div(log_p, q, reduction="batchmean")
        return mse + kl
    
    @staticmethod
    def _contrastive_loss_single_dim(self,
                                     q: Tensor,
                                     p: Tensor,
                                     target: Tensor,
                                     reduction: str = 'mean') -> Tensor:
        """InfoNCE contrastive loss for a single dimension slice."""
        q = F.normalize(q, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)
        logits = torch.matmul(q, p.transpose(0, 1)) / self.temperature
        return F.cross_entropy(logits, target, reduction=reduction)
    
    def learn_to_express(self,
                         layer_x: List[Tensor],
                         layer_y: List[Tensor],
                         target: Tensor,
                         reduction: str = 'mean') -> Tensor:
        """
        Eq. 2: L_le = sum_{i=1}^{n-1} w_i * loss(X^k_i, G) + loss(X^k_n, G)
        Applied across nested_dims (Matryoshka-style).
        
        layer_x: list of n tensors (batch, d) — query embeddings per layer
        layer_y: list of n tensors (batch, d) — positive embeddings per layer
        """
        n = len(layer_x)
        full_dim = layer_x[0].size(-1)
        total_loss = 0.0
        n_terms = 0

        for dim in self.nested_dims:
            if dim > full_dim:
                break

            for i in range(n):
                q_k = layer_x[i][:, :dim]
                p_k = layer_y[i][:, :dim]
                q_k = F.normalize(q_k, p=2, dim=1)
                p_k = F.normalize(p_k, p=2, dim=1)
                loss = self._contrastive_loss_single_dim(q_k, p_k, target, reduction)

                if i < n - 1:
                    w = self.layer_weight(i + 1)
                    total_loss = total_loss + w * loss
                else:
                    total_loss = total_loss + loss

            n_terms += 1

        if self.average_loss and n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss
    
    def learn_to_compress(self,
                          layer_x: List[Tensor]) -> Tensor:
        """
        Eq. 6: L_lc = sum_{i=1}^{n-1} w_i * align(X^k_i, X^k_i_pca) + align(X^k_n, X^k_n_pca)
        Applied across nested_dims (Matryoshka-style).
        
        layer_x: list of n tensors (batch, d) — query embeddings per layer
        """
        n = len(layer_x)
        full_dim = layer_x[0].size(-1)
        total_loss = 0.0
        n_terms = 0

        for dim in self.nested_dims:
            if dim > full_dim:
                break

            for i in range(n):
                emb = layer_x[i]
                # PCA target — detached, serves as soft label
                X_k_pca = self.pca_compress(emb, dim)
                # Truncated sub-embedding — learnable
                X_k_trunc = emb[:, :dim]

                a_loss = self.align_loss(X_k_trunc, X_k_pca)

                if i < n - 1:
                    w = self.layer_weight(i + 1)
                    total_loss = total_loss + w * a_loss
                else:
                    total_loss = total_loss + a_loss

            n_terms += 1

        if self.average_loss and n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss
    
    def __call__(self,
                 layer_x: List[Tensor],
                 layer_y: List[Tensor],
                 target: Tensor = None,
                 reduction: str = 'mean') -> Tensor:
        """
        L = α * L_le + β * L_lc
        
        Args:
            layer_x: list of n tensors (batch, d) — query embeddings per layer
            layer_y: list of n tensors (batch, d) — positive embeddings per layer
            target: (batch,) contrastive targets. Auto-generated if None.
        """
        assert len(layer_x) == len(layer_y), "layer_x and layer_y must have same number of layers"
        assert all(x.dim() == 2 for x in layer_x), "All layer embeddings must be 2D"

        device = layer_x[0].device

        if target is None:
            bs = layer_x[0].size(0)
            target_per_qry = layer_y[0].size(0) // bs
            target = torch.arange(0, bs * target_per_qry, target_per_qry, device=device, dtype=torch.long)

        l_le = self.learn_to_express(layer_x, layer_y, target, reduction)
        l_lc = self.learn_to_compress(layer_x)

        total = self.alpha * l_le + self.beta * l_lc
        return total
    
class DistributedESELoss(ESELoss):
    def __init__(self,
                 temperature: float = 0.02, 
                 nested_dims: List[int] = (64, 128, 256, 512, 1024),
                 alpha: float = 1.0, 
                 beta: float = 1.0,
                 average_loss: bool = False,
                 scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature, nested_dims, alpha, beta, average_loss)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.scale_loss = scale_loss
        
    def gather_tensor(self, t: Tensor) -> Tensor:
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
    
    def gather_layer_list(self, layers: List[Tensor]) -> List[Tensor]:
        return [self.gather_tensor(layer) for layer in layers]
    
    def __call__(self, 
                 layer_x: List[Tensor],
                 layer_y: List[Tensor],
                 **kwargs) -> Tensor:
        dist_layer_x = self.gather_layer_list(layer_x)
        dist_layer_y = self.gather_layer_list(layer_y)
        
        loss = super().__call__(dist_layer_x, dist_layer_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss