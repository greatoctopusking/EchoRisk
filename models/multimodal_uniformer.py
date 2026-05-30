import os
import torch
import torch.nn as nn

from models.uniformer import uniformer_small, uniformer_base


class MLPFusion(nn.Module):
    def __init__(self, dim=512, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2.bias.data[0] = 55.6

    def forward(self, f_a4c, f_a2c):
        x = torch.cat([f_a4c, f_a2c], dim=-1)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


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

        self.fusion = MLPFusion(dim=encoder_dim)

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

        f_a4c = self.encoder(a4c_video).view(batch_size, -1)
        f_a2c = self.encoder(a2c_video).view(batch_size, -1)

        null = self.null_emb.expand(batch_size, -1)
        f_a4c = torch.where(a4c_mask.unsqueeze(-1), f_a4c, null)
        f_a2c = torch.where(a2c_mask.unsqueeze(-1), f_a2c, null)

        return self.fusion(f_a4c, f_a2c)
