import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention,Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # add one dimension for batch unsqueeze(1)

class DitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        # mlp_ratio: hidden_size -> hidden_size * mlp_ratio used for pointwise feedforward
        super(DitBlock).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) # 不需要默认的自学习的偏移参数elementwise_affine=False
        self.attn = Attention(hidden_size, num_heads=num_heads,qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        ### pointwise feedforward
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # glue 计算量比较大，所以用近似glue
        approx_glue = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=approx_glue)
        # 生成6个偏移参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) # 分成6份，分割的为度为1
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1) # 内部实现 modulate
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return x
    
class FinalLayer(nn.Moudle):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

        def forward(self, x, c):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x
        
        

