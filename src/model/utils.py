import re
import torch
import numpy as np


# Implement Union-Find operator for constructing ui patches
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

def get_select_mask(tensor, skip_ratio=0, rand=False):
    # Use tensor operations for efficiency
    if type(tensor) == torch.Tensor:
        retain_mask = (tensor == -1).clone()
        unique_vals, counts = torch.unique(tensor, return_counts=True)

        for i, (val, count) in enumerate(zip(unique_vals, counts)):
            if val == -1:
                continue
            positions = (tensor == val).nonzero(as_tuple=True)[0]
            num_positions = len(positions)
            
            if num_positions == 1:
                retain_mask[positions] = True
            else:
                num_to_skip = int(round(num_positions * skip_ratio))
                num_to_retain = max(1, num_positions - num_to_skip)
                if rand:
                    # rand means random select subset of selective tokens for layer-wise
                    perm = torch.randperm(num_positions, device=tensor.device)
                    positions_to_retain = positions[perm[:num_to_retain]]
                else:
                    indices = torch.linspace(0, num_positions - 1, steps=num_to_retain).long()
                    positions_to_retain = positions[indices]
                    
                retain_mask[positions_to_retain] = True
    else:
        assert type(tensor) == np.ndarray
        retain_mask = (tensor == -1).copy()
        unique_vals, counts = np.unique(tensor, return_counts=True)

        for val, count in zip(unique_vals, counts):
            if val == -1:
                continue
            positions = np.nonzero(tensor == val)[0]
            num_positions = len(positions)
            
            if num_positions == 1:
                retain_mask[positions] = True
            else:
                num_to_skip = int(round(num_positions * skip_ratio))
                num_to_retain = max(1, num_positions - num_to_skip)
                if rand:
                    perm = np.random.permutation(num_positions)
                    positions_to_retain = positions[perm[:num_to_retain]]
                else:
                    indices = np.linspace(0, num_positions - 1, num=num_to_retain, dtype=int)
                    positions_to_retain = positions[indices]
                    
                retain_mask[positions_to_retain] = True
    return retain_mask

def parse_layer_type(str_ranges, L, default=0):
    # 0 is without layer token selection, 1 is with layer token selection
    result = [default] * L
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str_ranges)
    for start, end, value in matches:
        start, end, value = int(start) - 1, int(end) - 1, int(value)
        if end >= L:
            end = L - 1
        result[start:end + 1] = [value] * (end - start + 1)
    return result

def unnorm_pooling(self, last_hidden_state, attention_mask):
    if self.pooling == 'last' or self.pooling == 'eos':
        if getattr(self, "model_backbone", None) in ['qwen3_vl']:
            # print("Applying pooling for Qwen3VL backbone")
            flipped_tensor = attention_mask.flip(dims=[1])
            last_one_positions = flipped_tensor.argmax(dim=1)
            col = attention_mask.shape[1] - last_one_positions - 1
            row = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            return last_hidden_state[row, col]
        else:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                max_length = last_hidden_state.size(1)
                invert_mask = (attention_mask == 0).long()
                num_padding_tokens = invert_mask.sum(dim=1)
                eos_indices_positive = max_length - num_padding_tokens - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices_positive]
            return reps