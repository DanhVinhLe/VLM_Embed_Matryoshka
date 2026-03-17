from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveMatryoshkaStage1Loss(nn.Module):
    """
    Stage-1 loss for Adaptive Matryoshka representation learning.

    This implements:
      1) CLIP-style cross-modal alignment on a chosen student prefix.
      2) Curriculum distillation between adjacent prefixes.

    Supported prefix chain (default): [64, 256, 512, 1024].
    Curriculum phases:
      - A: align on full dim only.
      - B: align 512 + KL(sim_512 || sim_1024[detached]).
      - C: align 256 + KL(sim_256 || sim_512[detached]).
      - D: align 64  + KL(sim_64  || sim_256[detached]).
      - all: sum B+C+D style terms + full-dim alignment.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.02)
        nested_dims = getattr(args, "nested_dims", None) or [64, 256, 512, 768, 1024]
        self.nested_dims = sorted(set(nested_dims))
        self.phase = str(getattr(args, "stage1_phase", "all")).upper()
        self.distill_lambda = float(getattr(args, "distill_lambda", 0.5))

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
        return torch.cat(all_tensors, dim=0)

    def _build_contrastive_target(self, q: Tensor, p: Tensor) -> Tensor:
        # Supports grouped positives (n_hardneg + 1 layout used in this repo).
        target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        target_per_qry = p.size(0) // q.size(0)
        return target * target_per_qry

    def _project_to_dim(self, x: Tensor, dim: int) -> Tensor:
        """Project larger embedding dimensions into a smaller prefix dimension."""
        if x.size(-1) == dim:
            return x
        if x.size(-1) < dim:
            raise ValueError(f"Cannot project {x.size(-1)} -> {dim}: source dim is smaller.")

        # Research-friendly deterministic projection: feature-group average pooling.
        if x.size(-1) % dim == 0:
            group = x.size(-1) // dim
            return x.view(x.size(0), dim, group).mean(dim=-1)

        # Fallback for non-divisible dimensions.
        pooled = F.adaptive_avg_pool1d(x.unsqueeze(1), output_size=dim)
        return pooled.squeeze(1)

    def _similarity_logits(self, q: Tensor, p: Tensor, dim: int) -> Tensor:
        q_dim = F.normalize(self._project_to_dim(q, dim), p=2, dim=-1)
        p_dim = F.normalize(self._project_to_dim(p, dim), p=2, dim=-1)
        return (q_dim @ p_dim.t()) / self.temperature

    def _cross_alignment_l1(
        self,
        qry: Tensor,
        pos: Tensor,
        target: Tensor,
        dim: int,
        bigger_dim: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Cross alignment requested by review:
          - keep a single directional contrastive CE (like base_mrl.py)
          - add L1 between two cross-projected cosine similarity maps

        Branch A (contrastive + cosine map):
          qry_dim vs pos projected from bigger_dim -> dim.
        Branch B (cosine map only):
          qry projected from bigger_dim -> dim vs pos_dim.
        """
        if bigger_dim is None:
            bigger_dim = dim

        q_dim = F.normalize(self._project_to_dim(qry[:, :bigger_dim], dim), p=2, dim=-1)
        p_dim = F.normalize(self._project_to_dim(pos[:, :bigger_dim], dim), p=2, dim=-1)

        # One-direction contrastive CE, consistent with base_mrl style.
        logits = (q_dim @ p_dim.t()) / self.temperature
        contrastive = F.cross_entropy(logits, target)

        # Cross-projected cosine maps for L1 consistency.
        # Map-1: qry_dim x proj(pos_big)
        q_small = F.normalize(self._project_to_dim(qry, dim), p=2, dim=-1)
        p_from_big = F.normalize(self._project_to_dim(pos[:, :bigger_dim], dim), p=2, dim=-1)
        cosine_map_1 = q_small @ p_from_big.t()

        # Map-2: proj(qry_big) x pos_dim
        q_from_big = F.normalize(self._project_to_dim(qry[:, :bigger_dim], dim), p=2, dim=-1)
        p_small = F.normalize(self._project_to_dim(pos, dim), p=2, dim=-1)
        cosine_map_2 = q_from_big @ p_small.t()

        l1_consistency = F.l1_loss(cosine_map_1, cosine_map_2)
        return contrastive, l1_consistency, logits

    def _distill_kl(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """
        KL(P_teacher || P_student) with both sides from softmax distributions.
        Teacher logits are detached by caller.
        """
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        student_prob = F.softmax(student_logits, dim=-1)
        eps = 1e-12
        kl = teacher_prob * (torch.log(teacher_prob + eps) - torch.log(student_prob + eps))
        return kl.sum(dim=-1).mean()

    def _resolve_dims(self, full_dim: int) -> List[int]:
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)
        return sorted(set(valid_dims))

    def _resolve_selected_stage_ids(self, stage_pairs: List[Tuple[int, Optional[int]]]) -> List[int]:
        """
        Resolve user-selected curriculum stages.

        Backward compatibility:
          - "A/B/C/D" still maps to stage indices 0/1/2/3.
        Generalized behavior:
          - Any single alphabetic token maps to an index (A=0, B=1, ... Z=25).
          - Comma-separated lists are supported, e.g. "A,C" or "0,2,4".
          - "ALL" means include every available stage built from nested_dims.

        If selection is empty or invalid, defaults to [0] (largest/full dimension stage).
        """
        max_idx = len(stage_pairs) - 1
        phase = self.phase.strip().upper()

        if phase == "ALL":
            return list(range(len(stage_pairs)))

        tokens = [tok.strip() for tok in phase.replace(";", ",").split(",") if tok.strip()]
        selected_ids: List[int] = []
        for token in tokens:
            if token.isdigit():
                selected_ids.append(int(token))
                continue

            if token.isalpha():
                # Support arbitrary alphabetic stage labels beyond D.
                if len(token) == 1:
                    selected_ids.append(ord(token) - ord("A"))
                else:
                    # Accept labels like "PHASE_E" by reading the trailing letter.
                    last_char = token[-1]
                    if "A" <= last_char <= "Z":
                        selected_ids.append(ord(last_char) - ord("A"))

        selected_ids = sorted({idx for idx in selected_ids if 0 <= idx <= max_idx})
        return selected_ids or [0]

    def forward(self, model_trainer, input_data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        model = model_trainer.model
        qry_input = input_data["qry"]
        pos_input = input_data["pos"]

        qry_full = model.encode_input(qry_input)[0]
        pos_full = model.encode_input(pos_input)[0]

        if self.world_size > 1:
            qry_full = self._dist_gather_tensor(qry_full)
            pos_full = self._dist_gather_tensor(pos_full)

        full_dim = qry_full.size(-1)
        valid_dims = self._resolve_dims(full_dim)
        target = self._build_contrastive_target(qry_full, pos_full)

        # Build backbone-aware curriculum chain with full embedding dim as the first stage.
        # stage_pairs entries: (student_dim, teacher_dim_or_None)
        desc_dims = sorted(valid_dims, reverse=True)
        stage_pairs: List[Tuple[int, Optional[int]]] = []
        for idx, student_dim in enumerate(desc_dims):
            teacher_dim = None if idx == 0 else desc_dims[idx - 1]
            stage_pairs.append((student_dim, teacher_dim))

        selected_ids = self._resolve_selected_stage_ids(stage_pairs)

        losses = []
        align_losses = []
        distill_losses = []
        metrics: Dict[str, Tensor] = {}

        logits_cache: Dict[int, Tensor] = {}
        for dim in sorted({d for pair in stage_pairs for d in pair if d is not None}):
            logits_cache[dim] = self._similarity_logits(qry_full, pos_full, dim)

        for idx in selected_ids:
            student_dim, teacher_dim = stage_pairs[idx]
            align_ce, align_l1, student_logits = self._cross_alignment_l1(
                qry=qry_full,
                pos=pos_full,
                target=target,
                dim=student_dim,
                bigger_dim=teacher_dim,
            )
            align_loss = align_ce + align_l1

            if teacher_dim is None:
                total = align_loss
                distill_loss = torch.zeros_like(align_loss)
            else:
                teacher_logits = logits_cache[teacher_dim].detach()
                distill_loss = self._distill_kl(student_logits, teacher_logits)
                total = align_loss + self.distill_lambda * distill_loss

            metrics[f"align_ce_dim_{student_dim}"] = align_ce.detach()
            metrics[f"align_l1_dim_{student_dim}"] = align_l1.detach()
            metrics[f"align_loss_dim_{student_dim}"] = align_loss.detach()
            if teacher_dim is not None:
                metrics[f"distill_loss_{student_dim}_from_{teacher_dim}"] = distill_loss.detach()
            losses.append(total)
            align_losses.append(align_loss)
            distill_losses.append(distill_loss)

        final_loss = torch.stack(losses).mean()
        mean_align_loss = torch.stack(align_losses).mean()
        mean_distill_loss = torch.stack(distill_losses).mean()

        # Keep `contrastive_loss` for compatibility with existing trainer logging.
        metrics["loss"] = final_loss
        metrics["total_loss"] = final_loss.detach()
        metrics["contrastive_loss"] = mean_align_loss
        metrics["align_loss"] = mean_align_loss.detach()
        metrics["distill_loss"] = mean_distill_loss.detach()
        return metrics


class AdaptiveDimensionRouter(nn.Module):
    """
    Router MLP for Stage-2 adaptive dimension selection.

    Input  : query embedding (full dim).
    Output : logits over dimension levels [64, 256, 512, 1024] (or configured dims).
    """

    def __init__(self, input_dim: int, dim_levels: List[int], hidden_dim: int = 256):
        super().__init__()
        self.dim_levels = sorted(dim_levels)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.dim_levels)),
        )

    def forward(self, query_embedding: Tensor) -> Tensor:
        return self.mlp(query_embedding)


class AdaptiveRouterLoss(nn.Module):
    """
    Stage-2 loss for adaptive router training.

    - Builds pseudo labels by measuring retrieval correctness at each prefix and
      selecting the smallest dimension that reaches `router_accuracy_threshold`.
    - Optimizes CE(router_logits, target_dim_id) + alpha * expected_dimension_cost.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.02)
        self.dim_levels = sorted(getattr(args, "nested_dims", None) or [64, 256, 512, 768, 1024])
        self.alpha = float(getattr(args, "router_alpha", 0.01))
        self.threshold = float(getattr(args, "router_accuracy_threshold", 0.9))
        self.router_hidden_dim = int(getattr(args, "router_hidden_dim", 256))

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
        return torch.cat(all_tensors, dim=0)

    def _target_from_retrieval(self, q: Tensor, p: Tensor, target: Tensor) -> Tensor:
        # For each query, choose the smallest dim whose top-1 retrieval is correct.
        dim_levels = [d for d in self.dim_levels if d <= q.size(-1)]
        costs = torch.tensor(dim_levels, dtype=q.dtype, device=q.device)

        per_dim_correct = []
        for dim in dim_levels:
            qd = F.normalize(q[:, :dim], p=2, dim=-1)
            pd = F.normalize(p[:, :dim], p=2, dim=-1)
            logits = (qd @ pd.t()) / self.temperature
            pred = logits.argmax(dim=-1)
            correct = (pred == target)
            per_dim_correct.append(correct)

        correct_stack = torch.stack(per_dim_correct, dim=1)  # [bs, n_dim]
        enough = correct_stack.float() >= self.threshold

        # choose smallest valid idx; fallback to largest dim
        fallback = torch.full((q.size(0),), len(dim_levels) - 1, device=q.device, dtype=torch.long)
        has_hit = enough.any(dim=1)
        first_hit = enough.float().argmax(dim=1)
        target_idx = torch.where(has_hit, first_hit, fallback)
        return target_idx, costs

    def forward(self, model_trainer, input_data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        model = model_trainer.model
        qry = model.encode_input(input_data["qry"])[0]
        pos = model.encode_input(input_data["pos"])[0]

        if self.world_size > 1:
            qry = self._dist_gather_tensor(qry)
            pos = self._dist_gather_tensor(pos)

        target = torch.arange(qry.size(0), device=qry.device, dtype=torch.long)
        target_per_qry = pos.size(0) // qry.size(0)
        target = target * target_per_qry

        router = getattr(model, "router_head", None)
        if router is None:
            raise RuntimeError("Router head missing on model. Attach `model.router_head` before adaptive_router training.")
        router_logits = router(qry)
        target_dim_idx, dim_costs = self._target_from_retrieval(qry, pos, target)

        router_ce = F.cross_entropy(router_logits, target_dim_idx)

        probs = F.softmax(router_logits, dim=-1)
        normalized_costs = dim_costs / dim_costs.max().clamp_min(1.0)
        compute_penalty = (probs * normalized_costs.unsqueeze(0)).sum(dim=-1).mean()

        loss = router_ce + self.alpha * compute_penalty

        pred_idx = router_logits.argmax(dim=-1)
        acc = (pred_idx == target_dim_idx).float().mean()

        return {
            "loss": loss,
            # Keep `contrastive_loss` key for trainer compatibility; map it to total router objective.
            "contrastive_loss": loss,
            "router_total_loss": loss.detach(),
            "router_ce_loss": router_ce.detach(),
            "router_compute_penalty": compute_penalty.detach(),
            "router_acc": acc.detach(),
        }
