import os
import torch
import torch.nn as nn
from timm.layers import DropPath

from models.uniformer import uniformer_small, uniformer_base
from models.uniformer import conv_3x3x3, Attention, Mlp


class FusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2., drop_path=0.1):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=0.1)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class MultiModalEchoCoTr(nn.Module):
    def __init__(self, model_name='uniformer_small', pretrained=True, weights=None,
                 freeze_encoder_stages=0):
        super().__init__()
        self.model_name = model_name

        if model_name == 'uniformer_small':
            self.encoder = uniformer_small()
        elif model_name == 'uniformer_base':
            self.encoder = uniformer_base()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        if pretrained and weights is not None:
            print(f"[Pretrain] Loading weights from: {weights}")
            if not os.path.exists(weights):
                print(f"[Pretrain] WARNING: file not found: {weights}")
            else:
                file_size_mb = os.path.getsize(weights) / (1024 * 1024)
                print(f"[Pretrain] File exists, size: {file_size_mb:.1f} MB")
            state_dict = torch.load(weights, map_location='cpu', weights_only=True)
            print(f"[Pretrain] State dict has {len(state_dict)} keys")
            result = self.encoder.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"[Pretrain] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
            if result.unexpected_keys:
                print(f"[Pretrain] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")
            loaded = len(state_dict) - len(result.unexpected_keys) - len(result.missing_keys)
            print(f"[Pretrain] Loaded {loaded}/{len(state_dict)} keys "
                  f"({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected)")
        elif pretrained:
            print("[Pretrain] pretrained=True but weights is None. Training from scratch.")
        else:
            print("[Pretrain] pretrained=False. Training from scratch.")

        encoder_dim = self.encoder.embed_dim[-1]
        self.encoder.head = nn.Identity()

        self._freeze_stages(freeze_encoder_stages)

        self.concat_proj = nn.Conv3d(encoder_dim * 2, encoder_dim, 1)

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(dim=encoder_dim, num_heads=8, mlp_ratio=2., drop_path=0.1),
            FusionBlock(dim=encoder_dim, num_heads=8, mlp_ratio=2., drop_path=0.1),
        ])

        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.head[-1].bias.data[0] = 55.6

        self.null_emb = nn.Parameter(torch.zeros(1, encoder_dim))

    def _freeze_stages(self, num_stages):
        if num_stages < 1:
            return
        freezable = ['patch_embed1', 'blocks1', 'patch_embed2', 'blocks2',
                      'patch_embed3', 'blocks3', 'patch_embed4', 'blocks4']
        frozen_count = 0
        for i, name in enumerate(freezable):
            if i >= num_stages * 2:
                break
            if hasattr(self.encoder, name):
                module = getattr(self.encoder, name)
                if isinstance(module, nn.ModuleList):
                    for p in module.parameters():
                        p.requires_grad = False
                        frozen_count += 1
                else:
                    for p in module.parameters():
                        p.requires_grad = False
                        frozen_count += 1
        print(f"[Freeze] Frozen {frozen_count} parameters (stages ≤ {num_stages})")

    def forward(self, a4c_video, a2c_video, a4c_mask, a2c_mask):
        batch_size = a4c_video.shape[0]

        f_a4c = self.encoder.forward_features(a4c_video)
        f_a2c = self.encoder.forward_features(a2c_video)

        _, _, T, H, W = f_a4c.shape
        null_map = self.null_emb.reshape(1, -1, 1, 1, 1).expand(batch_size, -1, T, H, W)
        f_a4c = torch.where(a4c_mask.view(-1, 1, 1, 1, 1), f_a4c, null_map)
        f_a2c = torch.where(a2c_mask.view(-1, 1, 1, 1, 1), f_a2c, null_map)

        x = torch.cat([f_a4c, f_a2c], dim=1)
        x = self.concat_proj(x)

        for blk in self.fusion_blocks:
            x = blk(x)

        x = x.flatten(2).mean(-1)
        return self.head(x).squeeze(-1)
