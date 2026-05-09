import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict

from src.model.utils import unnorm_pooling
from .utils import count_clean_text_tokens, get_unpadded_hidden


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
    
    

    def create_semi_orthogonal_matrix(self, tensor):
        orig_dtype = tensor.dtype  # lưu dtype ban đầu (fp16)
        
        tensor = tensor.to(torch.float32)  # cast lên fp32

        rows, cols = tensor.shape
        if rows >= cols:
            q, _ = torch.linalg.qr(tensor, mode='reduced')
            w = q[:, :cols]
        else:
            q, _ = torch.linalg.qr(tensor, mode='reduced')
            w = q.T[:rows, :]

        w = w.to(orig_dtype)  # cast về lại fp16
        return w
    
    @torch.no_grad()
    def power_iteration_pca(
        self,
        X,
        r=256,
        power_iters=2,
        ns_iters=5,
        eps=1e-6,
    ):
        """
        PCA/subspace estimation bằng power iteration + Newton-Schulz.

        Args:
            X: [N, D] token/features đã flatten và remove padding.
            r: số principal directions.
            power_iters: số lần subspace iteration.
            ns_iters: số bước Newton-Schulz.
            eps: stability.

        Returns:
            Q: [D, r] orthonormal principal subspace basis.
        """
        N, D = X.shape

        # # Bước 0 (Khuyến nghị cho PCA thực sự): Trung bình hóa dữ liệu nếu X chưa được center
        # X = X - X.mean(dim=0, keepdim=True)

        # random init
        Q = torch.randn(D, r, device=X.device, dtype=X.dtype)
        Q = Q / (Q.norm(dim=0, keepdim=True) + eps)

        I = torch.eye(r, device=X.device, dtype=X.dtype)

        for _ in range(power_iters):
            # power iteration: Q <- X^T X Q
            Q = X.T @ (X @ Q)

            # ---------- orthogonalization ----------
            G = Q.T @ Q

            # Lưu lại norm để scale ngược lại sau Newton-Schulz
            g_norm = G.norm() + eps
            
            # stability normalization để đảm bảo hội tụ
            G_scaled = G / g_norm

            # Newton-Schulz inverse sqrt
            Y = G_scaled
            Z = I.clone()

            for _ in range(ns_iters):
                T = 0.5 * (3.0 * I - Z @ Y)
                Y = Y @ T
                Z = T @ Z

            # Bù lại hệ số scale đã chia ở trên: Z_true = Z / sqrt(g_norm)
            Z = Z / torch.sqrt(g_norm)

            # orthogonalize
            Q = Q @ Z

        return Q


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

        student_special_ids = torch.tensor(
            list(set(list(tokenizer.added_tokens_encoder.values()) +
                    tokenizer.all_special_ids)),
            device=device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_input_qry, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_input_pos, student_special_ids)

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        valid_qry_tokens = []
        valid_pos_tokens = []

        for i in range(bs):
            # 1. QUERY Processing
            if student_qry_image_features is not None and \
                cur_idx_qry_img < len(student_qry_image_features):
                # --- Vision ---
                stu_feat = student_qry_image_features[cur_idx_qry_img]
        
                # --- Text (Multimedia case) ---
                last_unpadded_hidden_state = get_unpadded_hidden(
                    student_qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    stu_feat.size(0),
                    student_input_qry['attention_mask'][i]
                )
                valid_qry_tokens.append(last_unpadded_hidden_state)
                cur_idx_qry_img += 1
            else:
                # --- Text Only case ---
                last_unpadded_hidden_state = get_unpadded_hidden(
                    student_qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    0,
                    student_input_qry['attention_mask'][i]
                )
                valid_qry_tokens.append(last_unpadded_hidden_state)

            # 2. POS Processing
            if student_pos_image_features is not None and \
                cur_idx_pos_img < len(student_pos_image_features):
                # --- Vision ---
                stu_feat = student_pos_image_features[cur_idx_pos_img]
        
                # --- Text (Multimedia case) ---
                last_unpadded_hidden_state = get_unpadded_hidden(
                    student_pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    stu_feat.size(0),
                    student_input_pos['attention_mask'][i]
                )
                valid_pos_tokens.append(last_unpadded_hidden_state)
                cur_idx_pos_img += 1
            else:
                # --- Text Only case ---
                last_unpadded_hidden_state = get_unpadded_hidden(
                    student_pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    0,
                    student_input_pos['attention_mask'][i]
                )
                valid_pos_tokens.append(last_unpadded_hidden_state)

        valid_qry_tokens = torch.cat(valid_qry_tokens, dim=0)  # [num_valid_tokens, hidden_dim]
        valid_pos_tokens = torch.cat(valid_pos_tokens, dim=0)  # [num_valid_tokens, hidden_dim]

        valid_full_tokens = torch.cat([valid_qry_tokens, valid_pos_tokens], dim=0)

        cnt = 0
        kd_loss = 0.0

        N, full_dim = valid_full_tokens.shape

        for i, dim in enumerate(nested_dims):
            if dim >= full_dim:
                continue
            
            # 1. Chốt số chiều an toàn (actual_dim)
            actual_dim = min(N, full_dim, dim)
            
            cnt += 1
            
            # ==========================================
            # LUỒNG 1: TẠO TEACHER TỪ FULL TOKENS
            # ==========================================
            Q_teacher = self.power_iteration_pca(valid_full_tokens, actual_dim)
            
            teacher_projected = valid_full_tokens @ Q_teacher

            # ==========================================
            # LUỒNG 2: TẠO STUDENT TỪ SLICED TOKENS
            # ==========================================
            # Cắt lấy không gian dim chiều
            student_sliced = valid_full_tokens[:, :dim]  # [N, dim]
            
            # DÙNG PCA CHIẾU SLICED TOKENS VỀ ACTUAL_DIM
            Q_student = self.power_iteration_pca(student_sliced, actual_dim)

            # Chiếu Sliced Tokens xuống actual_dim -> [N, actual_dim]
            student_projected = student_sliced @ Q_student

            # ==========================================
            # LUỒNG 3: CHUẨN HÓA VÀ TÍNH LOSS
            # ==========================================
            teacher_norm = F.normalize(teacher_projected, p=2, dim=-1)
            student_norm = F.normalize(student_projected, p=2, dim=-1)

            # Ép Student PCA Subspace học theo Teacher PCA Subspace
            loss_dim = F.mse_loss(student_norm, teacher_norm.detach())
            
            kd_loss += loss_dim

        if cnt > 0:
            kd_loss = kd_loss / cnt
        
        total_loss = contrastive_loss + self.kd_weight * kd_loss
        result = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }

        result.update(dim_losses)
        return result