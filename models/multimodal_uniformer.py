import torch
import torch.nn as nn

from models.uniformer import uniformer_small, uniformer_base


class MLPFusion(nn.Module):
    def __init__(self, dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.act = nn.GELU()
        self.fc3.bias.data[0] = 55.6

    def forward(self, f_a4c, f_a2c):
        x = torch.cat([f_a4c, f_a2c], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x).squeeze(-1)


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
            state_dict = torch.load(weights, map_location='cpu', weights_only=True)
            self.encoder.load_state_dict(state_dict, strict=False)

        encoder_dim = self.encoder.embed_dim[-1]
        self.encoder.head = nn.Identity()

        self.fusion = MLPFusion(dim=encoder_dim)

        self.null_emb = nn.Parameter(torch.zeros(1, encoder_dim))

    def forward(self, a4c_video, a2c_video, a4c_mask, a2c_mask):
        batch_size = a4c_video.shape[0]

        f_a4c = self.encoder(a4c_video).view(batch_size, -1)
        f_a2c = self.encoder(a2c_video).view(batch_size, -1)

        null = self.null_emb.expand(batch_size, -1)
        f_a4c = torch.where(a4c_mask.unsqueeze(-1), f_a4c, null)
        f_a2c = torch.where(a2c_mask.unsqueeze(-1), f_a2c, null)

        return self.fusion(f_a4c, f_a2c)
