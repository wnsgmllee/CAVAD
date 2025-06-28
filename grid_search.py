import json
import itertools
import torch
import numpy as np
import os
import random
import torch.nn as nn
import torch.optim as optim
from config import parser
from dataset import VADDataset
from model import CAM_GATSM, CAM_TSM, CAM_GATSM_R, CategoryAwareModule
from loss import TopkMILRankingLoss
from validate import validate
from transformers import get_linear_schedule_with_warmup
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    "CategoryAwareModule": CategoryAwareModule,
    "CAM_TSM": CAM_TSM,
    "CAM_GATSM": CAM_GATSM,
    "CAM_GATSM_R": CAM_GATSM_R
}

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.weight is not None and m.weight.dim() >= 2:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def load_hyperparams(path="hyperparams.json"):
    with open(path, "r") as f:
        return json.load(f)

def train_single_run(args):
    # Dataset
    anomaly_dataset = VADDataset(is_anomaly=True)
    normal_dataset = VADDataset(is_anomaly=False)

    anomaly_dataloader = data.DataLoader(anomaly_dataset, num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True, drop_last=True)
    normal_dataloader = data.DataLoader(normal_dataset, num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True, drop_last=True)

    sample = next(iter(anomaly_dataloader))
    input_feature = sample[0]
    input_dim = input_feature.shape[-1]
    num_categories = anomaly_dataset.num_categories

    ModelClass = model_dict[args.model]
    model = ModelClass(input_dim, num_categories, d_emb=args.d_emb, topk=args.topk, train=True).to(device)
    initialize_weights(model)

    mil_loss_fn = TopkMILRankingLoss(top_k=args.topk, lambda1=args.lambda_1, lambda2=args.lambda_2).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(anomaly_dataloader) * args.max_epoch
    warmup_steps = int(0.1 * total_steps)

    warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)

    best_auc = 0.0
    model.train()
    for epoch in range(args.max_epoch):
        for (normal_data, anomaly_data) in zip(normal_dataloader, anomaly_dataloader):
            normal_feat, _, normal_category_gt = normal_data
            anomaly_feat, anomaly_labels, anomaly_category_gt = anomaly_data

            normal_feat = normal_feat.to(device)
            anomaly_feat = anomaly_feat.to(device)
            anomaly_labels = anomaly_labels.to(device)
            normal_category_gt = normal_category_gt.to(device)
            anomaly_category_gt = anomaly_category_gt.to(device)

            optimizer.zero_grad()
            anomaly_scores, anomaly_category_logits = model(anomaly_feat, anomaly_category_gt)
            normal_scores, normal_category_logits = model(normal_feat, category_gt=normal_category_gt)

            loss_mil = mil_loss_fn(anomaly_scores, normal_scores)
            loss_category_anomaly = ce_loss_fn(anomaly_category_logits, torch.argmax(anomaly_category_gt, dim=1))
            loss_category_normal = ce_loss_fn(normal_category_logits, torch.argmax(normal_category_gt, dim=1))
            loss_category = args.lambda_category * (loss_category_anomaly + loss_category_normal)
            loss = loss_mil + loss_category
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()
        cosine_scheduler.step()

        visual_path = os.path.join(args.visual_root, args.visual_backbone)
        text_path = os.path.join(args.text_root, args.text_backbone)
        auc, ap = validate(model, args.test_list_path, args.gt_feature_path, visual_path, text_path)
        if auc > best_auc:
            best_auc = auc
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}, MIL: {loss_mil:.4f}, Cat: {loss_category:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    print(f"🔍 Final Best AUC for this config: {best_auc:.4f}")
    return best_auc

def run_all():
    hyper_grid = load_hyperparams()
    keys, values = zip(*hyper_grid.items())
    combos = list(itertools.product(*values))

    for i, combo in enumerate(combos):
        print(f"\n=== 🚀 RUN {i+1}/{len(combos)} ===")
        temp_args = parser.parse_args([])  # Load default config
        for key, val in zip(keys, combo):
            setattr(temp_args, key, val)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        auc = train_single_run(temp_args)

if __name__ == "__main__":
    run_all()
