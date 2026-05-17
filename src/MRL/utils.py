import numpy as np
import torch


def count_clean_text_tokens(inputs, special_ids_list):
    """
    Đếm số lượng token hợp lệ:
    1. Giá trị token phải >= 0 (loại bỏ -200, -100...)
    2. Giá trị token không nằm trong special_ids_list (loại bỏ CLS, SEP...)
    """
    input_ids = inputs['input_ids']
    
    if not isinstance(special_ids_list, torch.Tensor):
        # Nếu special_ids_list là list python thường, chuyển thành tensor
        special_ids_tensor = torch.tensor(special_ids_list, device=input_ids.device)
    else:
        # Nếu đã là tensor, đảm bảo cùng device
        special_ids_tensor = special_ids_list.to(input_ids.device)

    valid_index_mask = input_ids >= 0 
    content_mask = ~torch.isin(input_ids, special_ids_tensor)

    final_mask = valid_index_mask & content_mask

    return final_mask.sum(dim=1)

def get_hidden_text_vision(hidden_state, num_text_token, num_vision_token, attention_mask):
    '''
    Get hidden states for text and vision tokens separately
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        vision_hidden_state = hidden_state[-(num_vision_token+num_text_token): -num_text_token, :]
        text_hidden_state = hidden_state[-num_text_token:, :]
    else:
        vision_hidden_state = hidden_state[:num_vision_token, :]
        text_hidden_state = hidden_state[num_vision_token: num_vision_token + num_text_token, :]
   
    return text_hidden_state, vision_hidden_state

def get_unpadded_hidden(hidden_state, num_text_token, num_vision_token, attention_mask):
    '''
    Get hidden states for unpadded tokens (both text and vision)
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        num_vision_token: int, number of vision tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        unpadded_hidden_state = hidden_state[-(num_vision_token + num_text_token):, :]
    else:
        unpadded_hidden_state = hidden_state[: (num_vision_token + num_text_token), :]
   
    return unpadded_hidden_state

def get_hidden_text(hidden_state, num_text_token, attention_mask):
    '''
    Get hidden states for text tokens
    Args:
        hidden_state: tensor, the output hidden states from the model
        num_text_token: int, number of text tokens
        attention_mask: tensor, the attention mask indicating valid tokens # [Sequence length]
        (note: only )
    '''
    left_padding = attention_mask[0] == 0 and attention_mask[-1] == 1
    if left_padding:
        text_hidden_state = hidden_state[-num_text_token:, :]
    else:
        text_hidden_state = hidden_state[: num_text_token, :]
   
    return text_hidden_state


def get_full_attention_mask(hidden_state, num_text_token, num_vision_token, partial_attention_mask):
    '''
    Tạo ra full attention mask (1 cho token thật, 0 cho padding)
    Args:
        hidden_state: tensor shape [sequence_length, hidden_dim]
        num_text_token: int, số lượng text tokens
        num_vision_token: int, số lượng vision tokens
        partial_attention_mask: tensor mask cũ dùng để check padding direction
    Returns:
        full_mask: tensor shape [sequence_length] chứa mask đầy đủ
    '''
    # Lấy chiều dài chuỗi từ hidden_state
    seq_len = hidden_state.shape[0] 
    
    # Tổng số lượng token hợp lệ
    num_valid_tokens = num_vision_token + num_text_token
    
    # Khởi tạo mask mới toàn số 0 (coi như tất cả là padding ban đầu)
    full_mask = torch.zeros(seq_len, dtype=torch.long, device=hidden_state.device)
    
    # Xác định hướng padding từ mask đầu vào
    left_padding = partial_attention_mask[0] == 0 and partial_attention_mask[-1] == 1
    
    if left_padding:
        # Nếu left padding -> Token thật nằm ở CUỐI chuỗi
        full_mask[-num_valid_tokens:] = 1
    else:
        # Nếu right padding -> Token thật nằm ở ĐẦU chuỗi
        full_mask[:num_valid_tokens] = 1
        
    return full_mask


def get_text_only_attention_mask(hidden_state, num_text_token, num_vision_token, partial_attention_mask):
    '''
    Tạo ra mask CHỈ DÀNH CHO TEXT (1 cho text token, 0 cho vision token và padding)
    Args:
        hidden_state: tensor shape [sequence_length, hidden_dim]
        num_text_token: int, số lượng text tokens
        num_vision_token: int, số lượng vision tokens
        partial_attention_mask: tensor mask cũ dùng để check padding direction
    Returns:
        text_mask: tensor shape [sequence_length] chứa mask chỉ text
    '''
    seq_len = hidden_state.shape[0] 
    num_valid_tokens = num_vision_token + num_text_token
    
    # Khởi tạo mask mới toàn số 0 (coi như tất cả là padding/vision ban đầu)
    text_mask = torch.zeros(seq_len, dtype=torch.long, device=hidden_state.device)
    
    # Nếu câu không có text nào thì trả về luôn mask toàn 0
    if num_text_token == 0:
        return text_mask
        
    # Xác định hướng padding từ mask đầu vào
    left_padding = partial_attention_mask[0] == 0 and partial_attention_mask[-1] == 1
    
    if left_padding:
        # LEFT PADDING: [Padding...] -> [Vision Tokens...] -> [Text Tokens...]
        # Nghĩa là Text sẽ nằm gọn ở phần cuối cùng của chuỗi (last 'num_text_token' elements)
        text_mask[-num_text_token:] = 1
    else:
        # RIGHT PADDING: [Vision Tokens...] -> [Text Tokens...] -> [Padding...]
        # Bỏ qua đoạn đầu (thuộc về vision), bắt đầu từ 'num_vision_token'
        text_start_idx = num_vision_token
        text_end_idx = num_vision_token + num_text_token
        text_mask[text_start_idx:text_end_idx] = 1
        
    return text_mask