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
      2) Curriculum training across nested dimensions with trainable projections.
      3) Orthogonality regularization on each projection matrix (P^T P -> I).

    Supported prefix chain (default): [64, 128, 256, 512, 768, 1024].
    Curriculum phases:
      - A: align on full dim only.
      - B: align 512 using trainable projection(s).
      - C: align 256 using trainable projection(s).
      - D: align 64  using trainable projection(s).
      - all: sum all selected stages + orthogonality penalty.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.02)
        nested_dims = getattr(args, "nested_dims", None) or [64, 128, 256, 512, 768, 1024]
        self.nested_dims = sorted(set(nested_dims))
        self.phase = str(getattr(args, "stage1_phase", "all")).upper()
        self.teacher_source = str(getattr(args, "stage1_teacher_source", "previous")).strip().lower()
        if self.teacher_source not in {"previous", "full", "both"}:
            raise ValueError(
                f"Invalid stage1_teacher_source={self.teacher_source}. Use one of: previous, full, both"
            )
        self.align_l1_weight = float(getattr(args, "align_l1_weight", 1.0))
        self.full_dim_l1_weight = float(getattr(args, "full_dim_l1_weight", 0.0))
        self.orthogonal_weight = float(getattr(args, "orthogonal_weight", 0.01))
        self.orthogonal_pair_weights = self._parse_pair_weight_map(getattr(args, "orthogonal_pair_weights", ""))
        self.dim_align_l1_weights = self._parse_dim_weight_map(getattr(args, "align_l1_weights", ""))

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

    def _project_to_dim(self, model, x: Tensor, dim: int, src_dim: Optional[int] = None) -> Tensor:
        if src_dim is None:
            src_dim = x.size(-1)
        if src_dim == dim:
            return x[:, :dim]
        if src_dim < dim:
            raise ValueError(f"Cannot project {src_dim} -> {dim}: source dim is smaller.")
        if not hasattr(model, "matryoshka_proj_bank"):
            raise RuntimeError("Model missing `matryoshka_proj_bank`. Attach it before stage1 training.")
        return model.matryoshka_proj_bank.project(x[:, :src_dim], src_dim=src_dim, dst_dim=dim)

    def _cross_alignment_l1(
        self,
        model,
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

        q_dim = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_dim = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)

        # One-direction contrastive CE, consistent with base_mrl style.
        logits = (q_dim @ p_dim.t()) / self.temperature
        contrastive = F.cross_entropy(logits, target)

        # Cross-projected cosine maps for L1 consistency.
        # Use native prefix slices for the "small" side so only adjacent projections are required.
        # Map-1: qry_prefix(dim) x proj(pos_big->dim)
        q_small = F.normalize(qry[:, :dim], p=2, dim=-1)
        p_from_big = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)
        cosine_map_1 = q_small @ p_from_big.t()

        # Map-2: proj(qry_big->dim) x pos_prefix(dim)
        q_from_big = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_small = F.normalize(pos[:, :dim], p=2, dim=-1)
        cosine_map_2 = q_from_big @ p_small.t()

        l1_consistency = F.l1_loss(cosine_map_1, cosine_map_2)
        return contrastive, l1_consistency, logits

    def _resolve_dims(self, full_dim: int) -> List[int]:
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)
        return sorted(set(valid_dims))

    def _parse_dim_weight_map(self, spec) -> Dict[int, float]:
        """
        Parse per-dimension weight spec.

        Accepted format: "64:0.5,256:1.0,512:1.2"
        """
        if not spec:
            return {}
        if isinstance(spec, dict):
            return {int(k): float(v) for k, v in spec.items()}

        out: Dict[int, float] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"Invalid dim weight entry '{item}'. Expected format like '64:0.5,256:1.0'."
                )
            dim_str, weight_str = item.split(":", 1)
            out[int(dim_str.strip())] = float(weight_str.strip())
        return out

    def _resolve_selected_stage_ids(self, stage_pairs: List[Tuple[int, int]]) -> List[int]:
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

    def _parse_pair_weight_map(self, spec) -> Dict[Tuple[int, int], float]:
        """
        Parse per-pair orthogonal regularizer weights.

        Accepted format:
          - "1024->512:1.0,512->256:0.7"
          - "1024:512:1.0,512:256:0.7"
        """
        if not spec:
            return {}

        out: Dict[Tuple[int, int], float] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue

            if "->" in item and ":" in item:
                pair_spec, weight_spec = item.rsplit(":", 1)
                src_str, dst_str = pair_spec.split("->", 1)
            else:
                parts = item.split(":")
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid orthogonal pair weight entry '{item}'. "
                        f"Use '1024->512:1.0' (or '1024:512:1.0')."
                    )
                src_str, dst_str, weight_spec = parts

            src_dim = int(src_str.strip())
            dst_dim = int(dst_str.strip())
            out[(src_dim, dst_dim)] = float(weight_spec.strip())
        return out

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

        # Build strict adjacent chain:
        # full -> full/2 -> full/4 -> ...
        # For FastVLM configured dims this becomes 896->768->512->256->128->64.
        desc_dims = sorted(valid_dims, reverse=True)
        stage_pairs: List[Tuple[int, int]] = []
        for idx in range(len(desc_dims) - 1):
            teacher_dim = desc_dims[idx]
            student_dim = desc_dims[idx + 1]
            stage_pairs.append((teacher_dim, student_dim))

        selected_ids = self._resolve_selected_stage_ids(stage_pairs)

        losses = []
        align_losses = []
        orth_losses = []
        metrics: Dict[str, Tensor] = {}

        if not stage_pairs:
            # Single-dimension fallback (no projection pair available).
            align_ce, align_l1, _ = self._cross_alignment_l1(
                model=model,
                qry=qry_full,
                pos=pos_full,
                target=target,
                dim=desc_dims[0],
                bigger_dim=desc_dims[0],
            )
            weighted_align_loss = align_ce + self.full_dim_l1_weight * align_l1
            metrics["loss"] = weighted_align_loss
            metrics["total_loss"] = weighted_align_loss.detach()
            metrics["contrastive_loss"] = weighted_align_loss.detach()
            metrics["align_loss"] = weighted_align_loss.detach()
            metrics["orthogonal_loss"] = torch.zeros_like(weighted_align_loss).detach()
            return metrics

        for idx in selected_ids:
            teacher_dim, student_dim = stage_pairs[idx]
            align_ce, align_l1, _ = self._cross_alignment_l1(
                model=model,
                qry=qry_full,
                pos=pos_full,
                target=target,
                dim=student_dim,
                bigger_dim=teacher_dim,
            )

            l1_weight = self.dim_align_l1_weights.get(student_dim, self.align_l1_weight)
            weighted_align_loss = align_ce + l1_weight * align_l1

            if hasattr(model, "matryoshka_proj_bank"):
                base_orth = model.matryoshka_proj_bank.orthogonality_loss(src_dim=teacher_dim, dst_dim=student_dim)
                pair_weight = self.orthogonal_pair_weights.get((teacher_dim, student_dim), 1.0)
                orth_loss = pair_weight * base_orth
            else:
                orth_loss = torch.zeros_like(weighted_align_loss)

            total = weighted_align_loss + self.orthogonal_weight * orth_loss

            metrics[f"align_ce_{teacher_dim}_to_{student_dim}"] = align_ce.detach()
            metrics[f"align_l1_{teacher_dim}_to_{student_dim}"] = align_l1.detach()
            metrics[f"align_l1_weight_{teacher_dim}_to_{student_dim}"] = torch.tensor(l1_weight, device=align_ce.device)
            metrics[f"align_loss_{teacher_dim}_to_{student_dim}"] = weighted_align_loss.detach()
            metrics[f"orthogonal_loss_{teacher_dim}_to_{student_dim}"] = orth_loss.detach()
            losses.append(total)
            align_losses.append(weighted_align_loss)
            orth_losses.append(orth_loss)

        final_loss = torch.stack(losses).mean()
        mean_align_loss = torch.stack(align_losses).mean()
        mean_orth_loss = torch.stack(orth_losses).mean()

        # Keep `contrastive_loss` for compatibility with existing trainer logging.
        metrics["loss"] = final_loss
        metrics["total_loss"] = final_loss.detach()
        metrics["contrastive_loss"] = mean_align_loss
        metrics["align_loss"] = mean_align_loss.detach()
        metrics["orthogonal_loss"] = mean_orth_loss.detach()
        return metrics


class PairwiseProjectionBank(nn.Module):
    """Trainable projection matrices P for mapping src_dim -> dst_dim with orthogonality regularization."""

    def __init__(self, dimension_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.projections = nn.ParameterDict()
        for src_dim, dst_dim in dimension_pairs:
            key = self._key(src_dim, dst_dim)
            self.projections[key] = nn.Parameter(self._init_projection(src_dim, dst_dim))

    @staticmethod
    def _key(src_dim: int, dst_dim: int) -> str:
        return f"{int(src_dim)}_to_{int(dst_dim)}"

    @staticmethod
    def _init_projection(src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return torch.eye(src_dim, dtype=torch.float32)
        mat = torch.randn(src_dim, dst_dim, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q[:, :dst_dim]

    def project(self, x: Tensor, src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return x[:, :dst_dim]
        key = self._key(src_dim, dst_dim)
        if key not in self.projections:
            raise KeyError(f"Missing projection matrix for {src_dim}->{dst_dim}.")
        return x @ self.projections[key]

    def orthogonality_loss(self, src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return torch.zeros((), device=next(self.parameters()).device)
        key = self._key(src_dim, dst_dim)
        if key not in self.projections:
            raise KeyError(f"Missing projection matrix for {src_dim}->{dst_dim}.")
        p = self.projections[key]
        gram = p.transpose(0, 1) @ p
        eye = torch.eye(dst_dim, device=p.device, dtype=p.dtype)
        return ((gram - eye) ** 2).mean()


class AdaptiveDimensionRouter(nn.Module):
    """
    Router MLP for Stage-2 adaptive dimension selection.

    Input  : query embedding (full dim).
    Output : logits over dimension levels [64, 128, 256, 512, 768, 1024] (or configured dims).
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
        self.dim_levels = sorted(getattr(args, "nested_dims", None) or [64, 128, 256, 512, 768, 1024])
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
