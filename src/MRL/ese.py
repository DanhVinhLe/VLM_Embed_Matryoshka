from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Optional
import random

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
        total_loss, final_loss_dict, final_acc_dict = self._matryoshka_contrastive_loss(
            final_q, final_p, target,
        )
        all_metrics = {}
        all_metrics.update(final_loss_dict)
        all_metrics.update(final_acc_dict)

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

                layer_loss, layer_ld, layer_ad = self._matryoshka_contrastive_loss(
                    layer_q, layer_p, target,
                )

                # Deeper layers (closer to final) get higher weight
                layer_weight = 1.0 / (1.0 + math.log(num_layers - layer_idx))
                total_loss += layer_weight * layer_loss

                for k, v in layer_ld.items():
                    all_metrics[f"layer{layer_idx}_{k}"] = v
                for k, v in layer_ad.items():
                    all_metrics[f"layer{layer_idx}_{k}"] = v
                    
                del layer_q, layer_p, layer_loss
                    
        del qry_output, pos_output, qry_reps, pos_reps
        del qry_hidden_states, pos_hidden_states, final_q, final_p
        torch.cuda.empty_cache()

        return {
            'loss': total_loss,
            'contrastive_loss': total_loss,
            **all_metrics,
        }