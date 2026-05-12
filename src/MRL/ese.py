from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Optional
import random

from src.grad_cache import loss

class ESELoss(nn.Module):
    def __init__(self, args):
        super(ESELoss, self).__init__()
        self.args = args
        self.temperature = getattr(args, 'temperature', 0.02)
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.alpha = getattr(args, 'ese_alpha', 1.0)
        self.beta = getattr(args, 'ese_beta', 1.0)
        self.average_loss = getattr(args, 'average_loss', True)
        self.n_layers_per_step = getattr(args, 'n_layers_per_step', 0)  # 0 = use all layers
        self.kd_weight = getattr(args, 'kd_weight', 0.1)
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
    



    @torch.no_grad()
    def batched_ns_orthogonalize(self, Q, ns_iters=5, eps=1e-6):
        """
        Q: [B, D, r]
        return Q_orth: [B, D, r], Q^T Q ≈ I
        """
        B, D, r = Q.shape
        I = torch.eye(r, device=Q.device, dtype=Q.dtype).unsqueeze(0)  # [1, r, r]

        G = Q.transpose(1, 2) @ Q        # [B, r, r]

        g_norm = G.norm(dim=(1, 2), keepdim=True) + eps  # [B,1,1]
        G_scaled = G / g_norm

        Y = G_scaled
        Z = I.repeat(B, 1, 1)

        for _ in range(ns_iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

        Z = Z / torch.sqrt(g_norm)

        Q = Q @ Z                        # [B, D, r]
        return Q


    @torch.no_grad()
    def ese_batch_project_newton_schulz(
        self,
        x,
        r=128,
        power_iters=3,
        ns_iters=5,
        eps=1e-6,
    ):
        """
        ESE-style batch compression bằng Newton-Schulz.

        Args:
            x: [B, D]
            r: target dim

        Returns:
            z:  [B, r]      compressed target
            Ak: [B, D, r]   approximate U Sigma
            Q:  [B, D, r]   approximate top-left singular basis U
        """
        B, D = x.shape

        x_col = x.unsqueeze(-1)           # [B, D, 1]
        x_row = x.unsqueeze(1)            # [B, 1, D]

        A = torch.softmax(
            (x_col @ x_row) / math.sqrt(D),
            dim=-1,
        )                                # [B, D, D]
        Q = torch.randn(B, D, r, device=x.device, dtype=x.dtype)
        Q = Q / (Q.norm(dim=1, keepdim=True) + eps)

        for _ in range(power_iters):
            Q = A @ (A.transpose(1, 2) @ Q)        # [B, D, r]
            Q = self.batched_ns_orthogonalize(Q, ns_iters=ns_iters, eps=eps)

        ATQ = A.transpose(1, 2) @ Q                # [B, D, r]
        sigma = ATQ.norm(dim=1) + eps              # [B, r]

        Ak = Q * sigma.unsqueeze(1)                # [B, D, r] ≈ U Sigma

        z = Ak.transpose(1, 2) @ x.unsqueeze(-1)   # [B, r, 1]
        z = z.squeeze(-1)                          # [B, r]

        return z, Ak, Q

    
    def _matryoshka_contrastive_loss(
        self,
        emb1: Tensor,
        emb2: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Dict, Dict]:
        """
        EPRESSO-style Matryoshka contrastive loss for a single layer.
        Applies InfoNCE across multiple nested embedding dimensions.

        Args:
            emb1: [batch_size, full_dim] - query embeddings
            emb2: [N, full_dim] - key embeddings (N >= batch_size when gathered)
            target: [batch_size] - contrastive targets

        Returns:
            total_loss, loss_dict, acc_dict
        """
        full_dim = emb1.size(-1)
        device = emb1.device

        # Filter valid dims (<= full_dim), always include full_dim
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)

        # Log-based weights per dimension (like EPRESSO)
        dim_weights = [1.0 / (1.0 + math.log(i + 1)) for i in range(len(valid_dims))]

        total_loss = 0.0
        loss_dict = {}
        acc_dict = {}

        for idx, dim in enumerate(valid_dims):
            w = dim_weights[idx]

            # Slice to current dimension
            q = emb1[:, :dim]
            k = emb2[:, :dim]

            # Normalize embeddings
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

            # InfoNCE: similarity / temperature -> cross entropy
            logits = (q @ k.t()) / self.temperature
            loss = F.cross_entropy(logits, target)

            weighted_loss = w * loss
            total_loss += weighted_loss

            loss_dict[f"loss_dim_{dim}"] = loss.item()
            loss_dict[f"weighted_loss_dim_{dim}"] = weighted_loss.item()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean().item()
                acc_dict[f"acc_dim_{dim}"] = acc

        # Average across all dimension levels
        if self.average_loss and len(valid_dims) > 0:
            total_loss = total_loss / len(valid_dims)

        return total_loss, loss_dict, acc_dict
    
    def _learn_to_compress_loss(
        self,
        emb: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """
        Learn-to-compress loss without extra weights / hyperparameters.

        emb: [B, D]

        For each k in self.nested_dims:
            student_k = emb[:, :k]
            target_k  = no_grad(ESE-style PCA/Newton-Schulz projection emb -> k)
            loss      = MSE(student_k, target_k)

        Gradient only flows through student_k.
        PCA target is detached / no_grad.
        """
        full_dim = emb.size(-1)

        valid_dims = [d for d in self.nested_dims if d < full_dim]

        if len(valid_dims) == 0:
            return emb.new_tensor(0.0), {}
        
        dim_weights = [1.0 / (1.0 + math.log(i + 1)) for i in range(len(valid_dims))]

        total_loss = emb.new_tensor(0.0)
        metrics = {}

        # Normalize full embedding for stable compression target
        emb_full = F.normalize(emb, p=2, dim=-1)

        for i, dim in enumerate(valid_dims):
            # PCA / Newton-Schulz target: no gradient
            with torch.no_grad():
                target_k, _, _ = self.ese_batch_project_newton_schulz(
                    emb_full.detach(),
                    r=dim,
                )
                target_k = F.normalize(target_k, p=2, dim=-1)

            # First-k sub-embedding: has gradient
            student_k = emb_full[:, :dim]
            student_k = F.normalize(student_k, p=2, dim=-1)

            mse = F.mse_loss(student_k, target_k)

            kl = F.kl_div(
                F.log_softmax(student_k, dim=-1),
                F.softmax(target_k, dim=-1),
                reduction="batchmean",
            )
            loss = mse + kl
            total_loss = total_loss + dim_weights[i] * loss / 2

            metrics[f"compress_loss_dim_{dim}"] = loss.detach().item()

        return total_loss, metrics

    def forward(self, model_trainer, input_data):
        """
        EPRESSO-style loss: Matryoshka dimensions × layers.

        Final layer: full Matryoshka contrastive loss (weight = 1.0)
        Intermediate layers: optionally sampled, weighted by 1/(1+log(distance_from_top))
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

        # hidden_states: [0]=embedding layer, [1:]=transformer layers
        num_layers = len(qry_hidden_states)

        # ---- Helper: pool + gather ----
        def pool_and_gather(hidden_state, attn_mask):
            emb = self.eos_pooling(hidden_state, attn_mask)
            if self.world_size > 1:
                emb = self._dist_gather_tensor(emb)
            return emb

        # ---- Pool final layer & build contrastive target ----
        final_q = pool_and_gather(qry_hidden_states[-1], qry_attn_mask)
        final_p = pool_and_gather(pos_hidden_states[-1], pos_attn_mask)

        bs = final_q.size(0)
        target_per_qry = final_p.size(0) // bs
        target = torch.arange(
            0, bs * target_per_qry, target_per_qry,
            device=final_q.device, dtype=torch.long,
        )

        # ========== Final layer: full Matryoshka loss (weight=1.0) ==========
        final_loss, final_loss_dict, final_acc_dict = self._matryoshka_contrastive_loss(
            final_q, final_p, target,
        )

        lc_layer_loss_q, _ = self._learn_to_compress_loss(final_q)
        lc_layer_loss_p, _ = self._learn_to_compress_loss(final_p)

        final_lc_loss = (lc_layer_loss_q + lc_layer_loss_p) / 2

        all_metrics = {}
        all_metrics.update(final_loss_dict)
        all_metrics.update(final_acc_dict)

        kd_loss = final_lc_loss

        # ========== Intermediate layers ==========
        if num_layers > 2:
            # Exclude embedding layer [0] and final layer [-1]
            layer_indices = list(range(1, num_layers - 1))

            # Optionally sample a subset of intermediate layers
            if 0 < self.n_layers_per_step < len(layer_indices):
                layer_indices = random.sample(layer_indices, self.n_layers_per_step)

            for layer_idx in layer_indices:
                layer_q = pool_and_gather(qry_hidden_states[layer_idx], qry_attn_mask)
                layer_p = pool_and_gather(pos_hidden_states[layer_idx], pos_attn_mask)

                le_layer_loss, layer_ld, layer_ad = self._matryoshka_contrastive_loss(
                    layer_q, layer_p, target,
                )

                lc_layer_loss_q, _ = self._learn_to_compress_loss(layer_q)
                lc_layer_loss_p, _ = self._learn_to_compress_loss(layer_p)
                lc_layer_loss = (lc_layer_loss_q + lc_layer_loss_p) / 2

                # Deeper layers (closer to final) get higher weight
                layer_weight = 1.0 / (1.0 + math.log(layer_idx))
                kd_loss += layer_weight * (le_layer_loss + lc_layer_loss)

                for k, v in layer_ld.items():
                    all_metrics[f"layer{layer_idx}_{k}"] = v
                for k, v in layer_ad.items():
                    all_metrics[f"layer{layer_idx}_{k}"] = v
                    
                del layer_q, layer_p, le_layer_loss, lc_layer_loss, layer_ld, layer_ad
                    
        del qry_output, pos_output, qry_reps, pos_reps
        del qry_hidden_states, pos_hidden_states, final_q, final_p
        torch.cuda.empty_cache()
        total_loss = final_loss + self.kd_weight * kd_loss
        return {
            'loss': total_loss,
            'contrastive_loss': final_loss,
            'kd_loss': kd_loss,
            **all_metrics,
        }