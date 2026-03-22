import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict

class MatryoshkaERLoss(nn.Module):
    def __init__(self, args):
        super(MatryoshkaERLoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.average_loss = getattr(args, 'average_loss', True)
        
        self.alpha_ce = getattr(args, 'alpha_ce', 1.0) 
        self.beta_align = getattr(args, 'beta_align', 1.0) 
        self.gamma_er = getattr(args, 'gamma_er', 0.05) 

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_svd_and_er(self, x: Tensor):
        """
        Tính toán SVD và Effective Rank 1 lần duy nhất cho toàn bộ các bước.
        Trả về Full U, Full S (dùng để slice cắt PCA) và giá trị ER Loss.
        """
        bs, d = x.shape
        outer_product = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        A_d = F.softmax(outer_product / math.sqrt(d), dim=-1)

        # 1. TÍNH FULL SVD (Chỉ 1 lần)
        U, S, _ = torch.linalg.svd(A_d.float())
        
        U = U.to(x.dtype)
        S = S.to(x.dtype)

        # 2. TÍNH FULL EFFECTIVE RANK TỪ S
        eps = 1e-8
        p_full = S / (S.sum(dim=-1, keepdim=True) + eps)
        h_entropy = -torch.sum(p_full * torch.log(p_full + eps), dim=-1)
        
        effective_rank = (torch.exp(h_entropy) / d).mean()

        return U, S, effective_rank

    def project_pca(self, x: Tensor, U: Tensor, S: Tensor, dim: int):
        """
        Slice ma trận U và S tới kích thước dim, sau đó project x.
        Cực kỳ nhanh vì SVD đã được tính sẵn.
        """
        U_k = U[:, :, :dim]
        S_k = S[:, :dim]
        
        A_k = U_k * S_k.unsqueeze(1)
        x_pca = torch.bmm(A_k.transpose(1, 2), x.unsqueeze(2)).squeeze(2)
        return x_pca

    def align_loss(self, x_trunc: Tensor, x_pca: Tensor):
        """Hàm align (MSE + KLDiv)"""
        mse = F.mse_loss(x_trunc, x_pca)
        log_p = F.log_softmax(x_trunc, dim=-1)
        q = F.softmax(x_pca.detach(), dim=-1)
        kl = F.kl_div(log_p, q, reduction='batchmean')
        return mse + kl

    def forward(self, model_trainer, input_data):
        self.model_trainer = model_trainer
        model = model_trainer.model

        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        student_qry_reps = model.encode_input(student_input_qry)[0]
        student_pos_reps = model.encode_input(student_input_pos)[0]
        
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


        # ============================================================
        # BƯỚC 1: TIỀN XỬ LÝ SVD & EFFECTIVE RANK (Chỉ 1 lần duy nhất)
        # ============================================================
        U_qry, S_qry, q_er = self.compute_svd_and_er(all_student_qry_reps)
        U_pos, S_pos, p_er = self.compute_svd_and_er(all_student_pos_reps)
        
        er_loss = (q_er + p_er) / 2.0

        # ============================================================
        # BƯỚC 2: VÒNG LẶP MATRYOSHKA
        # ============================================================
        total_dim_loss = 0.0
        num_dims = 0

        for dim in self.nested_dims:
            if dim > full_dim: break
            
            # --- Contrastive Loss ---
            q_trunc = all_student_qry_reps[:, :dim]
            p_trunc = all_student_pos_reps[:, :dim]
            
            q_norm = F.normalize(q_trunc, p=2, dim=1)
            p_norm = F.normalize(p_trunc, p=2, dim=1)

            scores = model.compute_similarity(q_norm, p_norm)
            scores = scores.view(q_norm.size(0), -1)

            ce_loss = self.cross_entropy(scores / self.model_trainer.temperature, target)
            
            # --- Cắt Slice PCA Distillation ---
            if dim < full_dim:
                # Trích xuất và chiếu cực nhanh dựa trên U và S đã tính
                q_pca = self.project_pca(all_student_qry_reps, U_qry, S_qry, dim)
                p_pca = self.project_pca(all_student_pos_reps, U_pos, S_pos, dim)
                
                align_l = self.align_loss(q_trunc, q_pca) + self.align_loss(p_trunc, p_pca)
            else:
                align_l = torch.tensor(0.0, device=device)
            
            # --- Tổng hợp loss cho dimension hiện tại ---
            dim_loss = (self.alpha_ce * ce_loss) + (self.beta_align * align_l)
            total_dim_loss += dim_loss
            num_dims += 1

        # ============================================================
        # BƯỚC 3: TỔNG HỢP FINAL LOSS
        # ============================================================
        if self.average_loss and num_dims > 0:
            total_dim_loss = total_dim_loss / num_dims

        # Cộng ER Loss toàn cục
        final_total_loss = total_dim_loss + (self.gamma_er * er_loss)

        result = {
            "loss": final_total_loss,
            "contrastive_loss": ce_loss,
            "effective_rank_loss": er_loss
        }

        return result