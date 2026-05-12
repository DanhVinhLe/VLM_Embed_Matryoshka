import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict

from src.model.utils import unnorm_pooling
from .utils import count_clean_text_tokens, get_unpadded_hidden
import math

class SpreadLoss(nn.Module):
    def __init__(self, args):
        super(SpreadLoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.average_loss = getattr(args, 'average_loss', True)
        self.kd_weight = getattr(args, 'kd_weight', 0.1)
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.lambda_ortho = 1e-3

        self.nested_dims = sorted(self.nested_dims)
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    

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

        # ------------------------------------------------
        # 1. ESE inner-dependency matrix for whole batch
        # A_b = softmax(x_b x_b^T / sqrt(D))
        # ------------------------------------------------
        x_col = x.unsqueeze(-1)           # [B, D, 1]
        x_row = x.unsqueeze(1)            # [B, 1, D]

        A = torch.softmax(
            (x_col @ x_row) / math.sqrt(D),
            dim=-1,
        )                                # [B, D, D]

        # ------------------------------------------------
        # 2. Power iteration on A A^T
        # because A is not symmetric after row-softmax
        # ------------------------------------------------
        Q = torch.randn(B, D, r, device=x.device, dtype=x.dtype)
        Q = Q / (Q.norm(dim=1, keepdim=True) + eps)

        for _ in range(power_iters):
            Q = A @ (A.transpose(1, 2) @ Q)        # [B, D, r]
            Q = self.batched_ns_orthogonalize(Q, ns_iters=ns_iters, eps=eps)

        # Q ≈ U_r

        # ------------------------------------------------
        # 3. Approximate ESE A_k = U_r Sigma_r
        # sigma_i ≈ ||A^T u_i|| or ||A u_i||
        # Better for left singular vector u_i:
        # sigma_i = ||A^T u_i||
        # ------------------------------------------------
        ATQ = A.transpose(1, 2) @ Q                # [B, D, r]
        sigma = ATQ.norm(dim=1) + eps              # [B, r]

        Ak = Q * sigma.unsqueeze(1)                # [B, D, r] ≈ U Sigma

        # ------------------------------------------------
        # 4. ESE compressed embedding:
        # z = A_k^T x
        # ------------------------------------------------
        z = Ak.transpose(1, 2) @ x.unsqueeze(-1)   # [B, r, 1]
        z = z.squeeze(-1)                          # [B, r]

        return z, Ak, Q


    def forward(self, model_trainer, input_data):
        self.model_trainer = model_trainer
        model = model_trainer.model
        projectors = model_trainer.projectors
        processor = model_trainer.processor
        tokenizer = processor.tokenizer

        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        # Encode query and positive — get full-dim embeddings (unnormalized)
        student_qry_output = model.encode_input(student_input_qry)
        student_pos_output = model.encode_input(student_input_pos)

        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps

        bs, full_dim = all_student_qry_reps.shape
        device = all_student_qry_reps.device

        target = torch.arange(all_student_qry_reps.size(0), device=device, dtype=torch.long)
        target_per_qry = all_student_pos_reps.size(0) // all_student_qry_reps.size(0)
        target = target * target_per_qry

        contrastive_loss = 0.0
        num_dims = 0
        dim_losses = {}
        nested_dims = sorted(self.nested_dims)
        for dim in nested_dims:
            if dim > full_dim:
                break

            q = F.normalize(all_student_qry_reps[:, :dim], p=2, dim=1)
            p = F.normalize(all_student_pos_reps[:, :dim], p=2, dim=1)

            scores = model.compute_similarity(q, p)
            scores = scores.view(q.size(0), -1)

            loss = self.cross_entropy(scores / self.model_trainer.temperature, target)
            contrastive_loss += loss
            num_dims += 1
            dim_losses[f"contrastive_loss_dim_{dim}"] = loss.detach().item()

        if self.average_loss and num_dims > 0:
            contrastive_loss = contrastive_loss / num_dims

        cnt = 0
        kd_loss = 0.0

        full_rep = torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0)

        for dim in nested_dims:
            if dim >= full_dim:
                continue

            cnt += 1

            with torch.no_grad():
                target_k, _, _ = self.ese_batch_project_newton_schulz(
                    full_rep,
                    r=dim,
                    power_iters=3,
                    ns_iters=5,
                )   # [B, dim]
                target_k = F.normalize(target_k, p=2, dim=1)

            student_k = F.normalize(full_rep[:, :dim], p=2, dim=1)  # [B, dim]

            mse = F.mse_loss(student_k, target_k)

            kl = F.kl_div(
                F.log_softmax(student_k, dim=-1),
                F.softmax(target_k, dim=-1),
                reduction="batchmean",
            )

            loss_dim = mse + kl
            kd_loss = kd_loss + loss_dim

        if cnt > 0:
            kd_loss = kd_loss / cnt
        # print(f"KD Loss: {kd_loss.item():.4f}")
        total_loss = contrastive_loss + self.kd_weight * kd_loss
        result = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

        result.update(dim_losses)
        return result