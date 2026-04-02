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
