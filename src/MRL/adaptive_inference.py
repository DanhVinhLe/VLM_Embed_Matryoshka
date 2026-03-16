from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def route_and_truncate_query(
    model,
    processed_query: Dict[str, Tensor],
    dim_levels: List[int],
) -> Tuple[Tensor, int, Tensor]:
    """
    Inference helper for adaptive Matryoshka retrieval.

    Steps:
      query -> full embedding -> router prediction -> truncate embedding prefix.

    Returns:
      - truncated query embedding
      - selected dimension
      - router probability distribution over dimensions
    """
    full_query = model.encode_input(processed_query)[0]
    router = getattr(model, "router_head", None)
    if router is None:
        raise RuntimeError("Router head is required for adaptive inference but model.router_head is missing.")

    logits = router(full_query)
    probs = F.softmax(logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1)[0].item())
    valid_dims = sorted([d for d in dim_levels if d <= full_query.size(-1)])
    selected_dim = valid_dims[pred_idx]

    routed_query = F.normalize(full_query[:, :selected_dim], p=2, dim=-1)
    return routed_query, selected_dim, probs
