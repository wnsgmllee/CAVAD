import torch
import torch.nn as nn
from config import args


class SingleFeatureModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, args.proj_embedding) if input_dim != args.proj_embedding else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(args.proj_embedding, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.projection(x)
        scores = self.mlp(x).squeeze(-1)
        return scores



class MILVAD_Fusion(nn.Module):
    """📌 멀티모달 (Fusion) 모델"""
    def __init__(self, visual_dim, text_dim, fusion_type="concat"):
        super().__init__()
        self.fusion_type = fusion_type
        self.visual_proj = nn.Linear(visual_dim, args.proj_embedding)
        self.text_proj = nn.Linear(text_dim, args.proj_embedding)

        if fusion_type == "concat":
            fusion_dim = args.proj_embedding * 2
        elif fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(embed_dim=args.proj_embedding, num_heads=4, batch_first=True)
            fusion_dim = args.proj_embedding
        else:
            self.fusion_layer = nn.Linear(args.proj_embedding * 2, args.proj_embedding)
            fusion_dim = args.proj_embedding

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, visual_feat, text_feat):
        v_feat = self.visual_proj(visual_feat)
        t_feat = self.text_proj(text_feat)

        if self.fusion_type == "concat":
            fused_feat = torch.cat([v_feat, t_feat], dim=-1)
        elif self.fusion_type == "cross_attention":
            attn_output, _ = self.cross_attn(v_feat, t_feat, t_feat)
            fused_feat = attn_output
        else:
            fused_feat = self.fusion_layer(torch.cat([v_feat, t_feat], dim=-1))

        result = self.mlp(fused_feat).squeeze(-1)
        return result
