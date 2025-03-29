import torch 
import torch.nn as nn
import torch.nn.functional as F

def mil_loss(anomaly_scores, normal_scores, anomaly_labels, N_S=30):
    """
    📌 Top-K MIL Loss (Anomaly + Normal 활용)
    """
    batch_size = anomaly_scores.shape[0]
    loss_fn = nn.BCEWithLogitsLoss()

    video_scores = []

    for i in range(batch_size):
        top_scores, _ = torch.topk(anomaly_scores[i], min(N_S, anomaly_scores.shape[1])) # anomaly 비디오 내의 각 segment 별 이상 점수 중 상위 N_S개 선택
        video_score = torch.mean(top_scores).unsqueeze(0) # 해당 값 평균 계산 
        video_scores.append(video_score)

    normal_scores_mean = torch.mean(normal_scores, dim=1)
    normal_labels = torch.zeros_like(normal_scores_mean)

    video_scores = torch.cat([torch.cat(video_scores, dim=0), normal_scores_mean], dim=0) # normal 샘플 내의 각 semgment 별 이상 점수 평균 후 비디오 레벨의 점수 하나로 취급
    labels = torch.cat([anomaly_labels.squeeze(-1), normal_labels], dim=0)

    return loss_fn(video_scores, labels)
'''
def mil_loss(anomaly_scores, normal_scores, anomaly_labels, normal_labels, N_S=1): # upgraded mil_loss
    batch_size = anomaly_scores.shape[0]
    loss_fn = nn.BCEWithLogitsLoss()

    video_scores = []

    for i in range(batch_size):
        top_scores, _ = torch.topk(anomaly_scores[i], min(N_S, anomaly_scores.shape[1])) # anomaly 비디오 내의 각 segment 별 이상 점수 중 상위 N_S개 선택
        video_score = torch.mean(top_scores).unsqueeze(0) # 해당 값 평균 계산 
        video_scores.append(video_score)

    normal_scores_max = torch.max(normal_scores, dim=1).values # normal 샘플 내에서 가장 높은 이상 점수 선택
    normal_labels = torch.zeros_like(normal_scores_max)

    video_scores = torch.cat([torch.cat(video_scores, dim=0), normal_scores_max], dim=0) # normal 샘플에서 가장 높은 이상 점수를 비디오 레벨 점수로 취급
    labels = torch.cat([anomaly_labels.squeeze(-1), normal_labels], dim=0)

    return loss_fn(video_scores, labels)
'''

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anomaly_feats, normal_feats):
        """
        📌 Contrastive Loss (Anomaly vs. Normal 차이를 강화)
        """
        anomaly_feats = F.normalize(anomaly_feats, dim=-1)
        normal_feats = F.normalize(normal_feats, dim=-1)

        pos_sim = torch.exp(torch.sum(anomaly_feats * normal_feats, dim=-1) / self.temperature)
        neg_sim = torch.exp(torch.sum(anomaly_feats * anomaly_feats, dim=-1) / self.temperature)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss