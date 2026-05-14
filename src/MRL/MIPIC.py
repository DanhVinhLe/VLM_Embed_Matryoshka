import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional, Tuple
import math
from .utils import count_clean_text_tokens, get_full_attention_mask, get_unpadded_hidden

def Matry_infonce(a, b, temperature=0.07, nested_dims=[64, 128, 256, 512, 1024]):
    """
    Matryoshka InfoNCE loss - apply contrastive loss on nested dimensions
    
    Args:
        a: [batch_size, full_dim] tensor
        b: [batch_size, full_dim] tensor  
        temperature: temperature for InfoNCE
        nested_dims: list of dimensions to apply loss on
    
    Returns:
        total_loss: sum of losses across all nested dimensions
        all_logits: dict of logits for each dimension
    """
    # Debug: check input shapes
    assert a.dim() == 2, f"Expected a to be 2D [batch, dim], got shape {a.shape}"
    assert b.dim() == 2, f"Expected b to be 2D [batch, dim], got shape {b.shape}"
    assert a.shape == b.shape, f"a and b must have same shape, got {a.shape} vs {b.shape}"
    
    total_loss = 0.0
    all_logits = {}
    
    full_dim = a.size(1)
    
    for dim in nested_dims:
        if dim > full_dim:
            print(f"Warning: skipping dim={dim} as it exceeds full_dim={full_dim}")
            continue
            
        # Slice feature dimension, not batch dimension
        q = a[:, :dim]  # [batch_size, dim]
        k = b[:, :dim]  # [batch_size, dim]
        
        # Normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(q, k.T) / temperature  # [batch_size, batch_size]
        
        # Labels: diagonal should match (i-th query matches i-th key)
        labels = torch.arange(q.size(0), device=q.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        total_loss += loss
        
        all_logits[f'dim_{dim}'] = logits
    
    total_loss = total_loss / len(nested_dims) if nested_dims else torch.tensor(0.0)

    return total_loss, all_logits

class CKALoss(nn.Module):
    """
    CKA (Centered Kernel Alignment) Loss for measuring representation similarity
    
    Computes CKA per-example (using k tokens as samples) then averages over batch.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def compute_cka_single(self, X, Y):
        """
        Compute CKA for a single example
        
        Args:
            X: [n, p1] - n samples (tokens), p1 features
            Y: [n, p2] - n samples (tokens), p2 features
        
        Returns:
            CKA similarity score (scalar)
        """
        # Convert to float64 for numerical stability
        X = X.to(torch.float64)
        Y = Y.to(torch.float64)
        
        # Center the representations (zero mean across samples)
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        
        # Compute Gram matrices: K = XX^T, L = YY^T
        # Then compute HSIC using Frobenius norm
        # CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        
        XTY = X.t().matmul(Y)  # [p1, p2]
        XTX = X.t().matmul(X)  # [p1, p1]
        YTY = Y.t().matmul(Y)  # [p2, p2]
        
        numerator = torch.norm(XTY, 'fro') ** 2
        denominator = torch.norm(XTX, 'fro') * torch.norm(YTY, 'fro') + self.eps
        
        cka_sim = numerator / denominator
        
        return cka_sim

    def forward(self, SH, TH):
        """
        Args:
            SH: Student hidden states [B, k, d_s] or [k, d_s]
            TH: Teacher hidden states [B, k, d_t] or [k, d_t]
        
        Returns:
            CKA distance (1 - CKA similarity), averaged over batch
        """
        # Check if batched input
        if SH.dim() == 3:  # [B, k, d_s]
            B, k, d_s = SH.shape
            _, _, d_t = TH.shape
            
            # Compute CKA per-example
            cka_sims = []
            for i in range(B):
                cka_sim = self.compute_cka_single(SH[i], TH[i])  # Each is [k, d]
                cka_sims.append(cka_sim)
            
            # Average over batch
            avg_cka_sim = torch.stack(cka_sims).mean()
            
        elif SH.dim() == 2:  # [k, d_s] - single example
            avg_cka_sim = self.compute_cka_single(SH, TH)
            
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {SH.shape}")
        
        # Return distance (1 - similarity) for minimization
        return 1.0 - avg_cka_sim.float()


# ============================================================
# 2. HorizontalAttentionAlignment Module (FIXED)
# ============================================================
class VLMHorizontalAttentionAlignment(nn.Module):
    def __init__(self, d_small: int, d_full: int, d_att: int = 128, use_mlp=True):
        super().__init__()
        self.d_att = d_att
        
        # Custom Projector: Dùng MLP cho VLM để giữ feature tốt hơn, hoặc Linear thường
        if use_mlp:
            self.proj_small = nn.Sequential(
                nn.Linear(d_small, d_small // 2),
                nn.GELU(),
                nn.Linear(d_small // 2, d_att)
            )
            self.proj_full = nn.Sequential(
                nn.Linear(d_full, d_full // 2),
                nn.GELU(),
                nn.Linear(d_full // 2, d_att)
            )
        else:
            self.proj_small = nn.Linear(d_small, d_att)
            self.proj_full = nn.Linear(d_full, d_att)
            
        self.W_Q = nn.Linear(d_att, d_att)
        self.W_K = nn.Linear(d_att, d_att)

    def get_query_tokens(self, hidden_proj, attention_mask):
        """
        Lấy token đại diện (EOS/Pooling token) làm Query thay vì mặc định lấy index 0 như BERT.
        VLM thường dùng token hợp lệ cuối cùng.
        """
        batch_size = hidden_proj.shape[0]
        query_tokens = []
        
        for i in range(batch_size):
            mask_i = attention_mask[i]
            # Nếu là left padding (số 0 ở đầu, 1 ở cuối) -> token cuối là index -1
            if mask_i[0] == 0 and mask_i[-1] == 1:
                q_token = hidden_proj[i, -1, :]
            # Nếu là right padding -> token cuối nằm ở vị trí sum(mask) - 1
            else:
                last_valid_idx = mask_i.sum().item() - 1
                q_token = hidden_proj[i, last_valid_idx, :]
            query_tokens.append(q_token)
            
        return torch.stack(query_tokens, dim=0) # [B, d_att]

    def compute_attention_dist(self, hidden, proj, mask, temperature=1.0):
        # 1. Project xuống không gian d_att chung
        h_proj = proj(hidden)  # [B, L, d_att]
        
        # 2. Lấy Query (từ token đại diện) và Key (từ toàn bộ token)
        q_global = self.get_query_tokens(h_proj, mask) # [B, d_att]
        q = self.W_Q(q_global) # [B, d_att]
        k = self.W_K(h_proj)   # [B, L, d_att]
        
        # 3. Tính attention scores
        scores = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) # [B, L]
        scores = scores / math.sqrt(self.d_att)
        
        # 4. Masking: Gán -inf cho padding để softmax ra 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_probs = F.softmax(scores / temperature, dim=-1)
        return attn_probs, scores

    def forward(self, h_small, h_full, mask, temperature=1.0):
        small_probs, small_scores = self.compute_attention_dist(h_small, self.proj_small, mask, temperature)
        full_probs, full_scores = self.compute_attention_dist(h_full, self.proj_full, mask, temperature)
        
        # KL Divergence với epsilon tránh log(0)
        eps = 1e-8
        kl_loss = F.kl_div(
            (small_probs + eps).log(),
            (full_probs + eps),
            reduction='batchmean',
            log_target=False
        )
        return kl_loss, small_scores, full_scores


# ============================================================
# 3. SubmatrixCKALoss Module (FIXED)
# ============================================================
class SubmatrixCKALoss(nn.Module):
    """
    Aligns geometric structure of top-k important tokens using CKA.
    Uses per-dim attention scores for token selection (self-distillation).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.cka_loss = CKALoss(eps=eps)
    
    def select_top_k_tokens(
        self,
        hidden: torch.Tensor,  # [B, L, d]
        selection_scores: torch.Tensor,  # [B, L] - per-dim attention scores
        k: int,
        mask: Optional[torch.Tensor] = None  # [B, L]
    ) -> torch.Tensor:
        """
        Select top-k tokens based on per-dim attention scores.
        
        Args:
            hidden: Hidden states [B, L, d]
            selection_scores: Attention scores for THIS dimension [B, L]
            k: Number of tokens to select
            mask: Attention mask [B, L]
        
        Returns:
            selected_hidden: [B, k, d]
        """
        B, L, d = hidden.shape
        
        # Apply mask to scores (set padded to -inf)
        if mask is not None:
            scores_masked = selection_scores.masked_fill(mask == 0, float('-inf'))
        else:
            scores_masked = selection_scores
        
        # Get top-k indices
        _, top_k_indices = torch.topk(scores_masked, k=min(k, L), dim=1)  # [B, k]
        
        # Gather hidden states
        # Expand indices to match hidden dimension
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, d)  # [B, k, d]
        selected_hidden = torch.gather(hidden, 1, top_k_indices_expanded)  # [B, k, d]
        
        return selected_hidden
    
    def forward(
        self,
        h_small: torch.Tensor,  # [B, L, d_small]
        h_full: torch.Tensor,   # [B, L, d_full=1024]
        small_scores: torch.Tensor,  # [B, L] - per-dim attention scores for small
        k: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CKA loss on top-k token submatrices.
        Uses per-dim attention scores for selection.
        
        Returns:
            loss: 1 - CKA(small_sub, full_sub)
        """
        # Select top-k tokens using PER-DIM scores
        small_sub = self.select_top_k_tokens(h_small, small_scores, k, mask)  # [B, k, d_small]
        full_sub = self.select_top_k_tokens(h_full, small_scores, k, mask)  # [B, k, d_full]
        
        # Use your CKA loss implementation
        loss = self.cka_loss(small_sub, full_sub)
        
        return loss

class VLMPipelineInfoNCELoss(nn.Module):
    """
    Vertical information chaining cho VLM.
    Truyền thông tin từ layer nông (kích thước nhỏ) lên layer sâu (kích thước lớn).
    Sử dụng token đại diện (EOS/Pooling token) thay vì [CLS].
    """
    def __init__(self, d_src: int, d_tgt: int, d_hidden: int = 256):
        super().__init__()
        
        # Mạng chiếu phi tuyến tính (Projector) từ chiều nhỏ lên chiều lớn
        self.phi = nn.Sequential(
            nn.Linear(d_src, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_tgt)
        )
        
    def get_target_tokens(self, hidden_proj, attention_mask):
        """ Lấy token cuối cùng hợp lệ làm đại diện (giống logic của VLMHorizontal) """
        batch_size = hidden_proj.shape[0]
        query_tokens = []
        for i in range(batch_size):
            mask_i = attention_mask[i]
            if mask_i[0] == 0 and mask_i[-1] == 1: # Left padding
                q_token = hidden_proj[i, -1, :]
            else: # Right padding
                last_valid_idx = mask_i.sum().item() - 1
                q_token = hidden_proj[i, last_valid_idx, :]
            query_tokens.append(q_token)
        return torch.stack(query_tokens, dim=0)

    def forward(self, src_hidden, tgt_hidden, mask, temperature=0.07):
        # 1. Trích xuất token đại diện của 2 layer
        u_src = self.get_target_tokens(src_hidden, mask)  # [B, d_src]
        v_tgt = self.get_target_tokens(tgt_hidden, mask)  # [B, d_tgt]
        
        # 2. Chiếu (Project) layer nguồn và Stop-gradient layer đích
        u_proj = self.phi(u_src)  # [B, d_tgt]
        v_tgt = v_tgt.detach()    # QUAN TRỌNG: Ngăn layer sâu bị kéo ngược lại
        
        # 3. Chuẩn hóa
        u_proj = F.normalize(u_proj, dim=-1)
        v_tgt = F.normalize(v_tgt, dim=-1)
        
        # 4. Tính InfoNCE
        logits = torch.matmul(u_proj, v_tgt.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MIPICLoss(nn.Module):
    def __init__(self, args):
        super(MIPICLoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nested_dims = getattr(args, 'nested_dims', [64, 128, 256, 512, 1024])
        self.average_loss = getattr(args, 'average_loss', True)
        
        self.kd_weight = getattr(args, 'kd_weight', 1.0)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        self.nested_dims = sorted(self.nested_dims)  # Ensure nested_dims is sorted in descending order


        self.d_full = self.nested_dims[-1] # Sửa thành hidden_size model VLM của bạn
        self.d_att = getattr(args, 'd_att', 128)
        self.align_layers = getattr(args, 'align_layers', [-4, -8]) # Chọn index các layer cần align (vd: 2 layer cuối)
        self.alpha_attn = getattr(args, 'alpha_attn', 0.1)

        self.attn_alignments = nn.ModuleDict()
        for layer_idx in self.align_layers:
            for dim in self.nested_dims:
                if dim >= self.d_full: continue
                # Khởi tạo class ta vừa viết ở trên
                key = f"layer_{layer_idx}_dim_{dim}"
                self.attn_alignments[key] = VLMHorizontalAttentionAlignment(
                    d_small=dim, d_full=self.d_full, d_att=self.d_att, use_mlp=True
                )

        self.cka_module = SubmatrixCKALoss()
        
        # Cấu hình các bước nhảy truyền thông tin. 
        # Format: (Layer_Nguồn, Chiều_Nguồn, Layer_Đích, Chiều_Đích)
        # Ví dụ: Từ layer -12 (chiều 128) -> layer -8 (chiều 256) -> layer -4 (chiều 512)
        self.pic_dims = self.nested_dims
        num_pic_stages = len(self.pic_dims)
        
        # 2. Sinh tự động các layer tương ứng (pic_layers)
        # Thiết lập khoảng cách (stride) giữa các layer đặt trạm PIC. Mặc định là 4.
        pic_layer_stride = getattr(args, 'pic_layer_stride', 4)
        
        # Sinh danh sách index layer đếm ngược từ -1. 
        # Nếu có 5 stages và stride=4 -> [-1, -5, -9, -13, -17]
        pic_layers_reversed = [-(1 + i * pic_layer_stride) for i in range(num_pic_stages)]
        
        # Đảo ngược lại để đúng thứ tự luồng thông tin: từ nông (âm nhiều) đến sâu (âm ít)
        # Kết quả: [-17, -13, -9, -5, -1]
        self.pic_layers = list(reversed(pic_layers_reversed))
        
        # 3. Sinh tự động pipeline_pairs
        self.pipeline_pairs = []
        for i in range(num_pic_stages - 1):
            l_src = self.pic_layers[i]
            d_src = self.pic_dims[i]
            l_tgt = self.pic_layers[i + 1]
            d_tgt = self.pic_dims[i + 1]
            
            self.pipeline_pairs.append((l_src, d_src, l_tgt, d_tgt))
            
        # Debug (Có thể uncomment để kiểm tra lúc chạy)
        # print(f"Nested Dims: {self.pic_dims}")
        # print(f"Auto PIC Layers: {self.pic_layers}")
        # print(f"PIC Pipeline Pairs: {self.pipeline_pairs}")

        # 4. Khởi tạo ModuleDict
        self.pic_alignments = nn.ModuleDict()
        for l_src, d_src, l_tgt, d_tgt in self.pipeline_pairs:
            key = f"pic_{l_src}_{d_src}_to_{l_tgt}_{d_tgt}"
            self.pic_alignments[key] = VLMPipelineInfoNCELoss(
                d_src=d_src, 
                d_tgt=d_tgt, 
                d_hidden=max(d_src, d_tgt) // 2
            )
    

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
        processor = model_trainer.processor
        tokenizer = processor.tokenizer
        temperature = self.model_trainer.temperature

        student_input_qry = input_data['qry']
        student_input_pos = input_data['pos']

        # Encode query and positive — get full-dim embeddings (unnormalized)
        qry_output = model.encode_input(student_input_qry)
        pos_output = model.encode_input(student_input_pos)

        qry_reps, qry_image_features, qry_attn_matrix, qry_hidden_states = qry_output
        pos_reps, pos_image_features, pos_attn_matrix, pos_hidden_states = pos_output
        
        if self.world_size > 1:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_pos_reps = self._dist_gather_tensor(pos_reps)
        else:
            all_qry_reps = qry_reps
            all_pos_reps = pos_reps
        
        bs, full_dim = all_qry_reps.shape
        device = all_qry_reps.device
        
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

        qry_attention_mask = []
        pos_attention_mask = []

        for i in range(bs):
            # 1. QUERY Processing
            if qry_image_features is not None and \
                cur_idx_qry_img < len(qry_image_features):
                # --- Vision ---
                stu_feat = qry_image_features[cur_idx_qry_img]
        
                # --- Text (Multimedia case) ---
                qry_full_attention_mask = get_full_attention_mask(
                    qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    stu_feat.size(0),
                    student_input_qry['attention_mask'][i]
                )
                qry_attention_mask.append(qry_full_attention_mask)
                cur_idx_qry_img += 1
            else:
                # --- Text Only case ---
                qry_full_attention_mask = get_full_attention_mask(
                    qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    0,
                    student_input_qry['attention_mask'][i]
                )
                qry_attention_mask.append(qry_full_attention_mask)

            # 2. POS Processing
            if pos_image_features is not None and \
                cur_idx_pos_img < len(pos_image_features):
                # --- Vision ---
                stu_feat = pos_image_features[cur_idx_pos_img]
        
                # --- Text (Multimedia case) ---
                pos_full_attention_mask = get_full_attention_mask(
                    pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    stu_feat.size(0),
                    student_input_pos['attention_mask'][i]
                )
                pos_attention_mask.append(pos_full_attention_mask)
                cur_idx_pos_img += 1
            else:
                # --- Text Only case ---
                pos_full_attention_mask = get_full_attention_mask(
                    pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    0,
                    student_input_pos['attention_mask'][i]
                )
                pos_attention_mask.append(pos_full_attention_mask)

        qry_attention_mask = torch.stack(qry_attention_mask, dim=0)  # [B, L]
        pos_attention_mask = torch.stack(pos_attention_mask, dim=0)  # [B, L]

        contrastive_loss, _ = Matry_infonce(all_qry_reps, all_pos_reps, temperature=temperature, nested_dims=self.nested_dims)

        # SIA - Self-Distilled Intra-Relational Alignment
        # 2A Attention Distribution Matching
        total_attn_loss = 0.0
        total_cka_loss = 0.0  # Thêm biến theo dõi CKA loss
        align_count = 0
        
        # Base k để tính tỷ lệ top-k token (có thể truyền qua args)
        base_k = getattr(self.args, 'base_k', 64) 
        
        # Tính trên Query (thường self-distillation tính trên Query là đủ)
        for layer_idx in self.align_layers:
            # qry_hidden_states là tuple/list các tensor. Lấy tensor của layer cần tính
            layer_hidden = qry_hidden_states[layer_idx] # [B, L, d_full]
            current_dtype = layer_hidden.dtype

            for dim in self.nested_dims:
                if dim >= self.d_full: continue
                
                key = f"layer_{layer_idx}_dim_{dim}"
                attn_module = self.attn_alignments[key].to(device=device, dtype=current_dtype)
                
                # Cắt hidden state xuống d_small
                h_small = layer_hidden[..., :dim]
                h_full = layer_hidden
                
                # --------------------------------------------------
                # 2A: Attention Distribution Matching
                # --------------------------------------------------
                # QUAN TRỌNG: Lấy lại small_scores để dùng cho bước CKA
                kl_loss, small_scores, _ = attn_module(
                    h_small=h_small, 
                    h_full=h_full, 
                    mask=qry_attention_mask, 
                    temperature=temperature 
                )
                
                # --------------------------------------------------
                # 2B: Top-k Hidden State Alignment via CKA
                # --------------------------------------------------
                # Xác định số lượng k_i token dựa trên tỷ lệ kích thước dimension
                # Công thức: k_i = max(8, (dim / d_full) * base_k)
                k_i = max(8, int((dim / self.d_full) * base_k))
                
                self.cka_module = self.cka_module.to(device=device, dtype=current_dtype)

                cka_loss = self.cka_module(
                    h_small=h_small,
                    h_full=h_full,
                    small_scores=small_scores, # Dùng điểm Attention của chiều nhỏ làm định hướng
                    k=k_i,
                    mask=qry_attention_mask
                ).to(device)


                
                total_attn_loss += kl_loss
                total_cka_loss += cka_loss
                align_count += 1
                
        if align_count > 0:
            total_attn_loss = total_attn_loss / align_count
            total_cka_loss = total_cka_loss / align_count
        
        # ==========================================================
        # 2C: PIC - Progressive Information Chaining
        # ==========================================================
        total_pic_loss = 0.0
        pic_count = 0
        
        for l_src, d_src, l_tgt, d_tgt in self.pipeline_pairs:
            key = f"pic_{l_src}_{d_src}_to_{l_tgt}_{d_tgt}"
            current_dtype = qry_hidden_states[l_src].dtype
            
            # ---> SỬA Ở ĐÂY: Thêm dtype=current_dtype <---
            pic_module = self.pic_alignments[key].to(device=device, dtype=current_dtype)
            
            # Cắt hidden states đúng theo kích thước yêu cầu
            qry_src_hidden = qry_hidden_states[l_src][..., :d_src]
            qry_tgt_hidden = qry_hidden_states[l_tgt][..., :d_tgt]
            
            # Tính PIC loss
            pic_loss = pic_module(
                src_hidden=qry_src_hidden,
                tgt_hidden=qry_tgt_hidden,
                mask=qry_attention_mask,
                temperature=temperature # InfoNCE temperature (mặc định)
            )

            pos_src_hidden = pos_hidden_states[l_src][..., :d_src]
            pos_tgt_hidden = pos_hidden_states[l_tgt][..., :d_tgt]

            pic_loss += pic_module(
                src_hidden=pos_src_hidden,
                tgt_hidden=pos_tgt_hidden,
                mask=pos_attention_mask,
                temperature=temperature
            )
            
            total_pic_loss += pic_loss / 2
            pic_count += 1
            
        if pic_count > 0:
            total_pic_loss = total_pic_loss / pic_count

        # ==========================================================
        # 3. TỔNG HỢP FINAL LOSS
        # ==========================================================
        # Theo bài báo: L_SIA = L_att + L_CKA
        # Bạn có thể tách trọng số riêng cho CKA nếu muốn: alpha_attn và alpha_cka
        alpha_attn = getattr(self.args, 'alpha_attn', 0.1)
        alpha_cka = getattr(self.args, 'alpha_cka', 0.1)
        alpha_pic = getattr(self.args, 'alpha_pic', 0.1)

        sia_loss = (alpha_attn * total_attn_loss) + (alpha_cka * total_cka_loss)
        pic_final_loss = alpha_pic * total_pic_loss
        
        kd_loss = (sia_loss + pic_final_loss)
        # Loss cuối cùng
        final_loss = contrastive_loss + kd_loss * self.kd_weight

        result = {
            "loss": final_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "attn_kl_loss": total_attn_loss if align_count > 0 else torch.tensor(0.0),
            "cka_loss": total_cka_loss if align_count > 0 else torch.tensor(0.0),
            "pic_loss": total_pic_loss if pic_count > 0 else torch.tensor(0.0),
            "sia_loss": sia_loss if align_count > 0 else torch.tensor(0.0)
        }
        
        # result.update(dim_losses)

        return result