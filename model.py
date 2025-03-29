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


class CategoryAwareModule(nn.Module):
    """📌 Category Aware Module (CAM)"""
    def __init__(self, input_dim, cam_dim):
        super().__init__()
        self.feature_transform = nn.Linear(input_dim, 512)
        self.activation = nn.ReLU()
        self.feature_project = nn.Linear(512, cam_dim)

        # Weighted Aggregation (learnable weights instead of mean)
        self.aggregation_weights = nn.Linear(cam_dim, 1)

    def forward(self, features):
        """features: (batch, k, input_dim)"""
        transformed = self.activation(self.feature_transform(features))
        cam_features = self.activation(self.feature_project(transformed))  # (batch, k, cam_dim)

        # 🔥 Instead of mean, use learnable weighted sum
        attn_weights = torch.softmax(self.aggregation_weights(cam_features), dim=1)  # (batch, k, 1)
        cam_output = (attn_weights * cam_features).sum(dim=1)  # Weighted sum (batch, cam_dim)

        return cam_output


class MultiheadAttentionModule(nn.Module):
    """📌 Multihead Attention Module (Fixed)"""
    def __init__(self, cam_dim, fused_dim, embed_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(cam_dim, embed_dim)  # (batch, cam_dim) -> (batch, embed_dim)
        self.key_proj = nn.Linear(fused_dim, embed_dim)  # (batch, seq_len, fused_dim) -> (batch, seq_len, embed_dim)
        self.value_proj = nn.Linear(fused_dim, embed_dim)  # (batch, seq_len, fused_dim) -> (batch, seq_len, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, cam_feat, fused_feat):
        """ 
        cam_feat: (batch, cam_dim)  -> Query
        fused_feat: (batch, seq_len, fused_dim) -> Key, Value
        """
        batch_size, seq_len, _ = fused_feat.shape

        query = self.query_proj(cam_feat).unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, embed_dim)
        key = self.key_proj(fused_feat)  # (batch, seq_len, embed_dim)
        value = self.value_proj(fused_feat)  # (batch, seq_len, embed_dim)

        # ⚡ MultiheadAttention expects (seq_len, batch, embed_dim)
        query = query.transpose(0, 1)  # (seq_len, batch, embed_dim)
        key = key.transpose(0, 1)  # (seq_len, batch, embed_dim)
        value = value.transpose(0, 1)  # (seq_len, batch, embed_dim)

        attn_output, _ = self.multihead_attn(query, key, value)  # (seq_len, batch, embed_dim)

        return attn_output.transpose(0, 1)  # (batch, seq_len, embed_dim)


class MILVAD_Fusion(nn.Module):
    def __init__(self, visual_dim, text_dim, cam_dim=512, embed_dim=512, topk_ratio=0.1):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.cam_module = CategoryAwareModule(visual_dim, cam_dim)

        fused_dim = visual_dim + text_dim  
        self.attention_module = MultiheadAttentionModule(cam_dim, fused_dim, embed_dim)

        self.pre_mlp = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

        # 🔥 Fix: post_mlp input 차원을 embed_dim=512로 변경
        self.post_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),  # (batch, embed_dim=512) -> (batch, 256)
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, visual_feat, text_feat):
        """ 
        visual_feat: (batch_size, seq_len, visual_dim) or (seq_len, visual_dim) 
        text_feat: (batch_size, seq_len, text_dim) or (seq_len, text_dim)
        """
        # ✅ Batch 여부 확인
        is_batched = len(visual_feat.shape) == 3  # (batch_size, seq_len, visual_dim)
        
        if not is_batched:
            visual_feat = visual_feat.unsqueeze(0)  # (1, seq_len, visual_dim)
            text_feat = text_feat.unsqueeze(0)  # (1, seq_len, text_dim)
        
        scores = self.pre_mlp(visual_feat).squeeze(-1)  # (batch, seq_len)
        seq_len = visual_feat.shape[1]  # 시퀀스 길이
        
        # ✅ topk 값이 seq_len을 초과하지 않도록 조정
        k = min(max(1, int(self.topk_ratio * seq_len)), seq_len)  
        topk_indices = torch.topk(scores, k, dim=-1).indices  # (batch, k)

        # ✅ Gather 수행 시 차원 유지
        topk_visual_feat = torch.gather(
            visual_feat, 1, topk_indices.unsqueeze(-1).expand(-1, -1, visual_feat.shape[-1])
        )

        cam_feature = self.cam_module(topk_visual_feat)  
        fused_feat = torch.cat([visual_feat, text_feat], dim=-1)  
        attn_output = self.attention_module(cam_feature, fused_feat)  

        result = self.post_mlp(attn_output).squeeze(-1)  

        # ✅ 단일 샘플이면 다시 squeeze
        return result.squeeze(0) if not is_batched else result


'''
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
'''