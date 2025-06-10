import torch 
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from dataset import VADDataset
from model import CAM_GATSM
from loss import TopkMILRankingLoss
from config import args
from utils import save_model
from validate import validate, validate_and_plot
import glob
import random
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== Training Configuration =====")
print(f"Batch Size: {args.batch_size}")
print(f"Learning Rate: {args.lr}")
print(f"Max Epochs: {args.max_epoch}")
print(f"Lambda1 (smooth): {args.lambda_1}")
print(f"Lambda2 (sparse): {args.lambda_2}")
print(f"Num Segments / Video: {args.num_segments_per_video}")
print(f"Projection Embedding: {args.proj_embedding}")
print(f"Top-k: {args.topk}")
print(f"Lambda Category: {args.lambda_category}")
print(f"d_emb: {args.d_emb}")
print(f"Visual Backbone: {args.visual_backbone}")
print(f"Text Backbone: {args.text_backbone}")
print("=================================\n")

anomaly_dataset = VADDataset(is_anomaly=True)
normal_dataset = VADDataset(is_anomaly=False)

anomaly_dataloader = data.DataLoader(anomaly_dataset, num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True, drop_last=True)
normal_dataloader = data.DataLoader(normal_dataset, num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True, drop_last=True)

sample = next(iter(anomaly_dataloader))
input_feature = sample[0]
input_dim = input_feature.shape[-1]
num_categories = anomaly_dataset.num_categories

import torch.nn as nn

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.weight is not None and m.weight.dim() >= 2:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

model_dict = {
    "CategoryAwareModule": CategoryAwareModule,
    "CAM_TSM": CAM_TSM,
    "CAM_GATSM": CAM_GATSM,
    "CAM_GATSM_R": CAM_GATSM_R
}
ModelClass = model_dict[args.model]
if args.test_only:
    model = ModelClass(input_dim, num_categories, d_emb=args.d_emb, topk=args.topk, train=False).to(device)
else:
    model = ModelClass(input_dim, num_categories, d_emb=args.d_emb, topk=args.topk, train=True).to(device)
initialize_weights(model)

mil_loss_fn = TopkMILRankingLoss(top_k=args.topk, lambda1=args.lambda_1, lambda2=args.lambda_2).to(device)
ce_loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
total_steps = len(anomaly_dataloader) * args.max_epoch
warmup_steps = int(0.1 * total_steps)

warmup_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.max_epoch,
    eta_min=1e-6
)

if args.save_model:
    result_path = os.path.join("results", args.result_folder)
    os.makedirs(result_path, exist_ok=True)

def train():
    best_auc = 0.0
    model.train()
    for epoch in range(args.max_epoch):
        for batch_idx, ((normal_feat, _, normal_category_gt), (anomaly_feat, anomaly_labels, anomaly_category_gt)) in enumerate(zip(normal_dataloader, anomaly_dataloader)):
            normal_feat = normal_feat.to(device)
            anomaly_feat = anomaly_feat.to(device)
            anomaly_labels = anomaly_labels.to(device)
            normal_category_gt = normal_category_gt.to(device)
            anomaly_category_gt = anomaly_category_gt.to(device)

            optimizer.zero_grad()
            anomaly_scores, anomaly_category_logits = model(anomaly_feat, anomaly_category_gt)
            normal_scores, normal_category_logits = model(normal_feat, category_gt=normal_category_gt)
            loss_mil = mil_loss_fn(anomaly_scores, normal_scores)
            anomaly_category_target = torch.argmax(anomaly_category_gt, dim=1)
            normal_category_target = torch.argmax(normal_category_gt, dim=1)
            loss_category_anomaly = ce_loss_fn(anomaly_category_logits, anomaly_category_target)
            loss_category_normal = ce_loss_fn(normal_category_logits, normal_category_target)
            loss_category = args.lambda_category * (loss_category_anomaly + loss_category_normal)
            loss = loss_mil + loss_category 
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()
        cosine_scheduler.step()
        visual_feature_path = os.path.join(args.visual_root, args.visual_backbone)
        text_feature_path = os.path.join(args.text_root, args.text_backbone)
        overall_auc, overall_ap = validate(model, args.test_list_path, args.gt_feature_path, visual_feature_path, text_feature_path)
        print(f"Epoch {epoch+1}/{args.max_epoch}, Total_Loss: {loss:.4f}, MIL_Loss: {loss_mil:.4f}, Category_Loss: {loss_category:.4f}, AUC: {overall_auc:.4f}, AP: {overall_ap:.4f}")
        if overall_auc > best_auc:
            best_auc = overall_auc
            if args.save_model:
                save_model(model, os.path.join(result_path, "best_model.pth"))
        print("-" * 60)
    print(f"\nBest validation AUC for this config: {best_auc:.4f}")

if __name__ == "__main__":
    if args.test_only:
        weighted_model = validate_and_plot(model)
        visual_feature_path = os.path.join(args.visual_root, args.visual_backbone)
        text_feature_path = os.path.join(args.text_root, args.text_backbone)
        overall_auc, overall_ap = validate(model, args.test_list_path, args.gt_feature_path, visual_feature_path, text_feature_path)
        print(f"AUC: {overall_auc:.4f}, AP: {overall_ap:.4f}")
    else:
        train()
