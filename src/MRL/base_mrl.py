import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict


class MatryoshkaContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(MatryoshkaContrastiveLoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.average_loss = getattr(args, 'average_loss', True)
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

        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        # Encode query and positive — get full-dim embeddings (unnormalized)
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

        total_loss = 0.0
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
            total_loss += loss
            num_dims += 1
            dim_losses[f"contrastive_loss_dim_{dim}"] = loss.item()

        if self.average_loss and num_dims > 0:
            total_loss = total_loss / num_dims

        result = {
            "loss": total_loss,
            "contrastive_loss": total_loss,
        }
        result.update(dim_losses)

        return result