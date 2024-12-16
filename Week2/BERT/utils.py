import torch

def percentage(batch_size: int, max_index: int, current_index: int):
    # Calcuate epoch progress percentage

    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)

def mlm_accuracy(result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor):
    # Calculate MLM accuracy between ONLY masked words

    r = result.argmax(-1).masked_select(~inverse_token_mask)
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)

def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    # Calculate NSP accuracy between two tensors
    s = (result.argmax(1) == target.argmax(1)).sum()    
    return round(float(s / result.size(0)), 2)