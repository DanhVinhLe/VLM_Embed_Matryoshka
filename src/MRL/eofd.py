import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict
from .utils import count_clean_text_tokens, get_unpadded_hidden


class EOFDLoss(nn.Module):
    def __init__(self, args):
        super(EOFDLoss, self).__init__()
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
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def get_orthogonal_loss(self, projectors):
        ortho_loss = 0.0
        num_linear_layers = 0
        
        for proj_name, projector_seq in projectors.items():
            for module in projector_seq.modules():
                if isinstance(module, nn.Linear):
                    W = module.weight
                    out_dim, in_dim = W.shape
                    
                    if out_dim <= in_dim:
                        w_w_t = torch.matmul(W, W.t())
                        identity = torch.eye(out_dim, device=W.device, dtype=W.dtype)
                        # Dùng MSE để lấy trung bình trên từng phần tử ma trận
                        ortho_loss += F.mse_loss(w_w_t, identity)
                    else:
                        w_t_w = torch.matmul(W.t(), W)
                        identity = torch.eye(in_dim, device=W.device, dtype=W.dtype)
                        ortho_loss += F.mse_loss(w_t_w, identity)
                    
                    num_linear_layers += 1
                    
        # Lấy trung bình trên tổng số lớp Linear
        if num_linear_layers > 0:
            ortho_loss = ortho_loss / num_linear_layers
            
        return ortho_loss

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

        std_qry = valid_qry_tokens.std(dim=0, unbiased=False) # [hidden_dim]
        std_pos = valid_pos_tokens.std(dim=0, unbiased=False)

        # 1. Xác định hằng số power (p) theo bài báo (họ thường dùng 0.5 hoặc 1.0)
        power = 0.5

        # 2. Tính giá trị trung bình của các std
        mean_std_qry = std_qry.mean()
        mean_std_pos = std_pos.mean()

        # 3. Chuẩn hóa (chia std của từng chiều cho giá trị trung bình)
        eps = 1e-8
        std_scaled_qry = std_qry / (mean_std_qry + eps)
        std_scaled_pos = std_pos / (mean_std_pos + eps)

        # 4. Nâng lên lũy thừa p để tạo trọng số tập trung (EOFD weights)
        weight_qry = std_scaled_qry ** power
        weight_pos = std_scaled_pos ** power   

        weight_qry = weight_qry.unsqueeze(0)  # [1, hidden_dim]
        weight_pos = weight_pos.unsqueeze(0)  # [1, hidden_dim]   

        # valid_q_norm = F.normalize(valid_qry_tokens.float(), p=2, dim=1)
        # valid_p_norm = F.normalize(valid_pos_tokens.float(), p=2, dim=1)

        # valid_q_norm_detach = valid_q_norm.detach()
        # valid_p_norm_detach = valid_p_norm.detach()

        valid_q_detach = valid_qry_tokens.detach()
        valid_p_detach = valid_pos_tokens.detach()

        cnt = 0
        kd_loss = 0.0
        for i, dim in enumerate(nested_dims):
            if dim >= full_dim:
                continue
            cnt += 1
            adjacent_dim = nested_dims[i+1] if i < len(nested_dims) - 1 else dim
            q = projectors[f'{dim}_{adjacent_dim}'](valid_qry_tokens[:, :dim])
            p = projectors[f'{dim}_{adjacent_dim}'](valid_pos_tokens[:, :dim])

            # q_norm = F.normalize(q, p=2, dim=1)
            # p_norm = F.normalize(p, p=2, dim=1)

            qry_diff = weight_qry[:adjacent_dim] * (valid_q_detach - q).abs()
            pos_diff = weight_pos[:adjacent_dim] * (valid_p_detach - p).abs()
            weighted_squared_diff = (qry_diff.mean() + pos_diff.mean()) * 0.5  # [num_valid_tokens, hidden_dim]
            kd_loss += weighted_squared_diff

        if cnt > 0:
            kd_loss = kd_loss / cnt

        orthogonal_loss = self.get_orthogonal_loss(projectors)  # Tính loss orthogonal và có thể thêm vào kd_loss nếu muốn

        total_loss = contrastive_loss + self.kd_weight * kd_loss + self.lambda_ortho * orthogonal_loss

        result = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "orthogonal_loss": orthogonal_loss
        }
        result.update(dim_losses)

        return result