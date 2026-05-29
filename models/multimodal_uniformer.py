import torch
import torch.nn as nn

from models.uniformer import uniformer_small, uniformer_base


class GatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 2)

    def forward(self, f_a4c, f_a2c, a4c_mask, a2c_mask):
        w = self.gate(torch.cat([f_a4c, f_a2c], dim=-1))
        w = w.softmax(dim=-1)
        w[:, 0] = w[:, 0] * a4c_mask.float()
        w[:, 1] = w[:, 1] * a2c_mask.float()
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        return w[:, 0:1] * f_a4c + w[:, 1:2] * f_a2c


class MultiModalEchoCoTr(nn.Module):
    def __init__(self, model_name='uniformer_small', pretrained=True, weights=None):
        super().__init__()
        self.model_name = model_name

        if model_name == 'uniformer_small':
            self.encoder = uniformer_small()
        elif model_name == 'uniformer_base':
            self.encoder = uniformer_base()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        if pretrained and weights is not None:
            state_dict = torch.load(weights, map_location='cpu')
            self.encoder.load_state_dict(state_dict)

        encoder_dim = self.encoder.embed_dim[-1]
        self.encoder.head = nn.Identity()

        self.fusion = GatedFusion(dim=encoder_dim)
        self.head = nn.Linear(encoder_dim, 1)
        self.head.bias.data[0] = 55.6

        self.null_emb = nn.Parameter(torch.zeros(1, encoder_dim))

    def _encode_view(self, video):
        if video is None:
            return None
        return self.encoder(video).squeeze(-1)

    def forward(self, a4c_video, a2c_video, a4c_mask, a2c_mask):
        batch_size = a4c_video.shape[0] if a4c_video is not None else a2c_video.shape[0]

        f_a4c = self._encode_view(a4c_video)
        f_a2c = self._encode_view(a2c_video)

        if f_a4c is None:
            f_a4c = self.null_emb.expand(batch_size, -1)
        if f_a2c is None:
            f_a2c = self.null_emb.expand(batch_size, -1)

        fused = self.fusion(f_a4c, f_a2c, a4c_mask, a2c_mask)
        return self.head(fused).squeeze(-1)
