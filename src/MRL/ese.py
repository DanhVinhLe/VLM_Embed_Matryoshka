from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Optional

class ESELoss(nn.Module):
    def __init__(self, args):
        super(ESELoss, self).__init__()
        self.args = args
        self.temperature = getattr(args, 'temperature', 0.02)
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.alpha = getattr(args, 'ese_alpha', 1.0)
        self.beta = getattr(args, 'ese_beta', 1.0)
        self.average_loss = getattr(args, 'average_loss', True)
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def eos_pooling(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size = hidden_state.size(0)
        device = hidden_state.device
        
        left_padding = (attention_mask[:, -1].sum() == batch_size)
        if left_padding:
            return hidden_state[:, -1, :]
        
        max_length = hidden_state.size(1)
        num_padding_tokens = (attention_mask == 0).long().sum(dim=1)
        eos_indices = max_length - num_padding_tokens - 1
        row = torch.arange(batch_size, device=device)
        # normalize eos embeddings to prevent large variance across layers
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
        return hidden_state[row, eos_indices]
    
    def _pool_hidden_layers(self, hidden_states, attention_mask) -> List[Tensor]:
        transformers_layers = hidden_states[1:]
        pooled = []
        for h in transformers_layers:
            pooled.append(self.eos_pooling(h, attention_mask))
        return pooled
        
    def layer_weight(self, i : int) -> float:
        # w_i = 1.0 / (1.0 + math.log(i)), i is 1-index of layer, starting from 1
        return 1.0 /(1.0 + math.log(i))
    
    @torch.no_grad()
    def pca_compress(self, embeddings: Tensor, k: int) -> Tensor:
        """
        PCA-based compression using svd_lowrank (much faster than full SVD).
        embeddings: (batch, d)
        k: target dimension
        Returns: (batch, k) — detached
        """
        d = embeddings.size(-1)
        if k >= d:
            return embeddings.detach()

        orig_dtype = embeddings.dtype
        m = embeddings.float()
        A = F.softmax(m.T @ m / (d ** 0.5), dim=-1)  # (d, d)
        A = A.to(torch.float32)
        u, s, _ = torch.svd_lowrank(A, q=k)           # u: (d, k), s: (k,)
        topk_deps = u @ torch.diag(s)                  # (d, k)
        result = m @ topk_deps                           # (batch, k)
        return result.to(orig_dtype).detach()
    
    def align_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """align(x, y) = MSE(x, y) + KLDiv(x, y). y is detached soft label."""
        # x_norm = F.normalize(x, p=2, dim=-1)
        # y_norm = F.normalize(y, p=2, dim=-1)
        mse = F.mse_loss(x, y)
        log_p = F.log_softmax(x, dim=-1)
        q = F.softmax(y, dim=-1)
        kl = nn.KLDivLoss(reduction='batchmean')(log_p, q)
        return mse + kl
    
    def _contrastive_loss_single_dim(self,
                                     q_k: Tensor,
                                     p_k: Tensor,
                                     target: Tensor,
                                     reduction: str = 'mean') -> Tensor:
        """InfoNCE contrastive loss for a single dimension slice."""
        logits = (q_k @ p_k.t()) / self.temperature
        return F.cross_entropy(logits, target, reduction=reduction)
    
    def learn_to_express(self,
                        layer_x: List[Tensor],
                        layer_y: List[Tensor],
                        target: Tensor,
                        reduction: str = 'mean') -> Tensor:
        """
        Eq. 2: L_le = Σ_{i=1}^{n-1} w_i * loss(X^k_i, G) + loss(X^k_n, G)
        Applied to both k-dim sub-embeddings AND full-dim embeddings.
        """
        n = len(layer_x)
        full_dim = layer_x[0].size(-1)
        total_loss = 0.0
        n_terms = 0

        # 1) Process nested k-dimensional sub-embeddings
        for dim in self.nested_dims:
            if dim >= full_dim:  # Skip if dim >= full_dim
                continue

            dim_loss = 0.0
            for i in range(n):
                q_k = layer_x[i][:, :dim]
                p_k = layer_y[i][:, :dim]
                loss = self._contrastive_loss_single_dim(q_k, p_k, target, reduction)

                if i < n - 1:
                    w = self.layer_weight(i + 1)
                    dim_loss += w * loss
                else:
                    dim_loss += loss

            total_loss += dim_loss
            n_terms += 1

        # 2) Process FULL-dimensional embeddings (d-dim)
        full_loss = 0.0
        for i in range(n):
            q_full = layer_x[i]  # Full dimension
            p_full = layer_y[i]
            loss = self._contrastive_loss_single_dim(q_full, p_full, target, reduction)

            if i < n - 1:
                w = self.layer_weight(i + 1)
                full_loss += w * loss
            else:
                full_loss += loss

        total_loss += full_loss
        n_terms += 1

        # Average across all dimension levels
        if self.average_loss and n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss / n

    def learn_to_compress(self, layer_x: List[Tensor]) -> Tensor:
        """
        Eq. 6: L_lc = Σ_{i=1}^{n-1} w_i * align(X^k_i, X^k_i_pca) + align(X^k_n, X^k_n_pca)
        Applied to both k-dim sub-embeddings AND full-dim embeddings.
        """
        n = len(layer_x)
        full_dim = layer_x[0].size(-1)
        total_loss = 0.0
        n_terms = 0

        # 1) Process nested k-dimensional compressions
        for dim in self.nested_dims:
            if dim >= full_dim:  # Skip if dim >= full_dim
                continue

            dim_loss = 0.0
            for i in range(n):
                emb = layer_x[i]
                X_k_pca = self.pca_compress(emb, dim)
                X_k_trunc = emb[:, :dim]
                a_loss = self.align_loss(X_k_trunc, X_k_pca)

                if i < n - 1:
                    w = self.layer_weight(i + 1)
                    dim_loss += w * a_loss
                else:
                    dim_loss += a_loss

            total_loss += dim_loss
            n_terms += 1

        # 2) Process FULL-dimensional compression (d-dim)
        # Align full embedding with its PCA reconstruction
        full_loss = 0.0
        for i in range(n):
            emb = layer_x[i]
            X_full_pca = self.pca_compress(emb, full_dim)
            a_loss = self.align_loss(emb, X_full_pca)

            if i < n - 1:
                w = self.layer_weight(i + 1)
                full_loss += w * a_loss
            else:
                full_loss += a_loss

        total_loss += full_loss
        n_terms += 1

        # Average across all dimension levels
        if self.average_loss and n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss / n
    
    def forward(self, model_trainer, input_data):
        """
        L = α * L_le + β * L_lc
        
        Args:
            layer_x: list of n tensors (batch, d) — query embeddings per layer
            layer_y: list of n tensors (batch, d) — positive embeddings per layer
            target: (batch,) contrastive targets. Auto-generated if None.
        """
        qry_input = input_data['qry']
        pos_input = input_data['pos']
        model = model_trainer.model

        # ---- Forward ----
        qry_output = model.encode_input(qry_input)
        pos_output = model.encode_input(pos_input)

        qry_reps, _, _, qry_hidden_states = qry_output
        pos_reps, _, _, pos_hidden_states = pos_output
        
        qry_attn_mask = qry_input.get('attention_mask', None)
        pos_attn_mask = pos_input.get('attention_mask', None)

        layer_x = self._pool_hidden_layers(qry_hidden_states, qry_attn_mask)
        layer_y = self._pool_hidden_layers(pos_hidden_states, pos_attn_mask)

        # Gather layer embeddings across GPUs
        if self.world_size > 1:
            layer_x = [self._dist_gather_tensor(lx) for lx in layer_x]
            layer_y = [self._dist_gather_tensor(ly) for ly in layer_y]

        # ---- ESE target ----
        bs = layer_x[0].size(0)
        target_per_qry = layer_y[0].size(0) // bs
        ese_target = torch.arange(
            0, bs * target_per_qry, target_per_qry,
            device=layer_x[0].device, dtype=torch.long
        )

        # ---- ESE losses ----
        l_le = self.learn_to_express(layer_x, layer_y, ese_target)
        l_lc = self.learn_to_compress(layer_x)
        ese_loss = self.alpha * l_le + self.beta * l_lc

        return {
            'loss': ese_loss,
            'contrastive_loss': ese_loss,
        }