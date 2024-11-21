from typing import Optional, List, Any
import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """interface for attention: self, and cross"""
    
    def __init__(self, query_dim: int, heads: int, dim_head: int, 
                 dropout: float = 0., bias: bool = True, cross_attention_dim: Optional[int] = None,
                 out_bias: bool = True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, 
                cond: Optional[torch.Tensor]=None, 
                attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        raise NotImplementedError

