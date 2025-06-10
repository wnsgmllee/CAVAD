import torch
import torch.nn as nn
from config import args
import torch.nn.functional as F

class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GlobalAdaptiveTSM(nn.Module):
    def __init__(self, input_dim=None, fold_div=2, num_heads=4, depth=2):
        super().__init__()
        self.fold_div = fold_div
        self.num_heads = num_heads
        self.depth = depth
        self.predictor = ShiftPredictor()
        self.input_dim = input_dim
        if input_dim is not None:
            self.transformer = TemporalTransformer(dim=input_dim, num_heads=num_heads, depth=depth)
        else:
            self.transformer = None

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, T, D = x.shape
        fold = D // self.fold_div
        out = torch.zeros_like(x)
        if self.transformer is None:
            self.transformer = TemporalTransformer(dim=D, num_heads=self.num_heads, depth=self.depth).to(x.device)
        shift_ratio = self.predictor(x).squeeze(-1)
        shift_channels = (shift_ratio * fold).long()
        for b in range(B):
            for t in range(T):
                shift_n = shift_channels[b, t].item()
                if t > 0 and shift_n > 0:
                    out[b, t, :shift_n] = x[b, t-1, :shift_n]
                if t < T - 1 and shift_n > 0:
                    out[b, t, shift_n:2*shift_n] = x[b, t+1, shift_n:2*shift_n]
                out[b, t, 2*shift_n:] = x[b, t, 2*shift_n:]
        out = self.transformer(out)
        if squeeze:
            out = out.squeeze(0)
        return out

class CAM_GATSM(nn.Module): 
    def __init__(self, input_dim, num_categories, d_emb, topk=args.topk, train=True):
        super().__init__()
        self.topk = topk
        self.num_categories = num_categories
        self.train_mode = train
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, args.proj_embedding),
            nn.BatchNorm1d(args.proj_embedding),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.decision_top = nn.Linear(args.proj_embedding, 1)
        self.category_classifier = nn.Sequential(
            nn.Linear(args.proj_embedding, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_categories)
        )
        self.category_embedding = nn.Embedding(num_categories, d_emb)
        self.key_proj = nn.Linear(args.proj_embedding, d_emb)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_emb, num_heads=4, batch_first=True)
        self.final_proj_to_input = nn.Linear(d_emb, input_dim)
        self.decision_final = nn.Linear(args.proj_embedding, 1)
        if self.train_mode:
            self.GlobalAdaptiveTSM = GlobalAdaptiveTSM(fold_div=args.fold_div)
        else:
            self.GlobalAdaptiveTSM = GlobalAdaptiveTSM(input_dim=input_dim, fold_div=args.fold_div)

    def forward(self, x, category_gt=None):
        B, N, _ = x.shape
        x = self.GlobalAdaptiveTSM(x)
        shared_feat = self.shared_backbone_forward(x)
        top_scores = self.decision_top(shared_feat).squeeze(-1)
        aggregated_feat = self.aggregate_topk_features(shared_feat, top_scores)
        category_logits = self.category_classifier(aggregated_feat)
        cat_emb = self.get_category_embedding(category_logits, category_gt)
        proj_feat = self.cross_attention_projection(shared_feat)
        cat_context = self.cross_attention_forward(proj_feat, cat_emb)
        final_features = proj_feat + cat_context
        final_features_input_dim = self.final_proj_to_input(final_features)
        final_features_proj = self.shared_backbone_forward(final_features_input_dim)
        segment_scores = self.decision_final(final_features_proj).squeeze(-1)
        return segment_scores, category_logits

    def shared_backbone_forward(self, x):
        B, N, _ = x.shape
        x_flat = x.view(-1, x.shape[-1])
        feat = self.shared_backbone(x_flat)
        return feat.view(B, N, -1)

    def aggregate_topk_features(self, shared_feat, top_scores):
        B, N, C = shared_feat.shape
        topk = min(self.topk, top_scores.shape[1])
        _, topk_idx = torch.topk(top_scores, topk, dim=1)
        batch_idx = torch.arange(B).unsqueeze(-1).expand(-1, topk).to(shared_feat.device)
        topk_feats = shared_feat[batch_idx, topk_idx]
        return topk_feats.mean(dim=1)

    def get_category_embedding(self, category_logits, category_gt=None):
        weights = F.softmax(category_logits, dim=1)
        emb_weights = self.category_embedding.weight[:self.num_categories]
        weighted_emb = weights.matmul(emb_weights)
        if self.training and category_gt is not None:
            gt_idx = torch.argmax(category_gt, dim=1)
            gt_emb = self.category_embedding(gt_idx)
            return gt_emb + weighted_emb
        else:
            return weighted_emb

    def cross_attention_projection(self, shared_feat):
        return self.key_proj(shared_feat)

    def cross_attention_forward(self, proj_feat, cat_emb):
        query = cat_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, proj_feat, proj_feat)
        return attn_out.expand(-1, proj_feat.size(1), -1)
