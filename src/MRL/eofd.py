import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict


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
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def forward(self, model_trainer, input_data):
        self.model_trainer = model_trainer
        model = model_trainer.model
        projectors = model_trainer.projectors

        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        # Encode query and positive — get full-dim embeddings (unnormalized)
        student_qry_output = model_trainer.encode_input(input_data)
        student_pos_output = model_trainer.encode_input(input_data)

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

        for dim in self.nested_dims:
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
        

        last_hidden_qry_token = student_qry_hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        last_hidden_pos_token = student_pos_hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        qry_mask = student_input_qry['attention_mask'].bool()
        pos_mask = student_input_pos['attention_mask'].bool()

        valid_qry_tokens = last_hidden_qry_token[qry_mask] # [num_valid_qry_tokens, hidden_dim]
        valid_pos_tokens = last_hidden_pos_token[pos_mask]

        std_qry = valid_qry_tokens.std(dim=0, unbiased=False) # [hidden_dim]
        std_pos = valid_pos_tokens.std(dim=0, unbiased=False)

        # 1. Xác định hằng số power (p) theo bài báo (họ thường dùng 0.5 hoặc 1.0)
        power = 0.5

        # 2. Tính giá trị trung bình của các std
        mean_std_qry = std_qry.mean()
        mean_std_pos = std_pos.mean()

        # 3. Chuẩn hóa (chia std của từng chiều cho giá trị trung bình)
        std_scaled_qry = std_qry / mean_std_qry
        std_scaled_pos = std_pos / mean_std_pos

        # 4. Nâng lên lũy thừa p để tạo trọng số tập trung (EOFD weights)
        weight_qry = std_scaled_qry ** power
        weight_pos = std_scaled_pos ** power

        weight_qry = weight_qry.unsqueeze(0)  # [1, hidden_dim]
        weight_pos = weight_pos.unsqueeze(0)  # [1, hidden_dim]


        cnt = 0
        kd_loss = 0.0
        for dim in self.nested_dims:
            if dim > full_dim:
                continue
            cnt += 1
            q = projectors[f'{dim}'](valid_qry_tokens[:, :dim])
            p = projectors[f'{dim}'](valid_pos_tokens[:, :dim])
            q = F.normalize(q, p=2, dim=1) # [b, d]
            p = F.normalize(p, p=2, dim=1) # [b, d]
            weighted_squared_diff = (weight_qry * (valid_qry_tokens - q) ** 2 + weight_pos * (valid_pos_tokens - p) ** 2) * 0.5  # [num_valid_tokens, hidden_dim]
            kd_loss += weighted_squared_diff.mean()

        if cnt > 0:
            kd_loss = kd_loss / cnt
        
        total_loss = contrastive_loss + self.kd_weight * kd_loss

        result = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss
        }
        result.update(dim_losses)

        return result