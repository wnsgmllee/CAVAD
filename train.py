import torch 
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from dataset import VADDataset
from model import MILVAD_Fusion, SingleFeatureModel
from loss import mil_loss, ContrastiveLoss
from config import args
from utils import save_model
from validate import validate, validate_and_plot
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Config 정보 출력
print("\n===== Training Configuration =====")
print(f"Mode: {args.mode}")
print(f"Batch Size: {args.batch_size}")
print(f"Learning Rate: {args.lr}")
print(f"Max Epochs: {args.max_epoch}")
print(f"Visual Backbone: {args.visual_backbone}")
print(f"Text Backbone: {args.text_backbone}")
print("=================================\n")

# ✅ 데이터 로드 (anomaly & normal 따로)
anomaly_dataset = VADDataset(mode=args.mode, is_anomaly=True)
normal_dataset = VADDataset(mode=args.mode, is_anomaly=False)

anomaly_dataloader = data.DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
normal_dataloader = data.DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# ✅ 모델 초기화
sample_batch = next(iter(anomaly_dataloader))
if args.mode == "fusion":
    visual_feature_batch, text_feature_batch = sample_batch[0]
    visual_dim, text_dim = visual_feature_batch.shape[-1], text_feature_batch.shape[-1]
    model = MILVAD_Fusion(visual_dim, text_dim).to(device)
else:
    input_dim = sample_batch[0].shape[-1]
    model = SingleFeatureModel(input_dim).to(device)

contrastive_loss_fn = ContrastiveLoss().to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)


# ✅ 결과 저장 경로 설정
if args.save_model:
    result_path = os.path.join("results", args.result_folder)
    os.makedirs(result_path, exist_ok=True)


def train():
    best_auc = 0.0
    model.train()
    iter_normal = iter(normal_dataloader)

    for epoch in range(args.max_epoch):
        epoch_loss = 0.0
        
        if args.mode == "fusion":
            for data, anomaly_labels in anomaly_dataloader:
                anomaly_vfeat, anomaly_tfeat = data
                anomaly_vfeat, anomaly_tfeat = anomaly_vfeat.to(device), anomaly_tfeat.to(device)
                anomaly_labels = anomaly_labels.to(device)

                try:
                    (normal_vfeat, normal_tfeat), _ = next(iter_normal)
                except StopIteration:
                    iter_normal = iter(normal_dataloader)
                    (normal_vfeat, normal_tfeat), _ = next(iter_normal)

                normal_vfeat, normal_tfeat = normal_vfeat.to(device), normal_tfeat.to(device)
                optimizer.zero_grad()

                anomaly_scores = model(anomaly_vfeat, anomaly_tfeat)
                normal_scores = model(normal_vfeat, normal_tfeat)

                loss_mil = mil_loss(anomaly_scores, normal_scores, anomaly_labels, args.num_segments_per_video)
                loss_contrastive = contrastive_loss_fn(anomaly_vfeat, normal_vfeat)

                loss = loss_mil + args.lambda_contrastive * loss_contrastive
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        else:
            for data, anomaly_labels in anomaly_dataloader:
                anomaly_feat = data
                anomaly_feat = anomaly_feat.to(device)
                anomaly_labels = anomaly_labels.to(device)

                try:
                    (normal_feat), _ = next(iter_normal)
                except StopIteration:
                    iter_normal = iter(normal_dataloader)
                    (normal_feat), _ = next(iter_normal)

                normal_feat = normal_feat.to(device)
                optimizer.zero_grad()

                anomaly_scores = model(anomaly_feat)
                normal_scores = model(normal_feat)

                loss_mil = mil_loss(anomaly_scores, normal_scores, anomaly_labels, args.num_segments_per_video)
                loss_contrastive = contrastive_loss_fn(anomaly_feat, normal_feat)

                loss = loss_mil + args.lambda_contrastive * loss_contrastive
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # ✅ Feature Path 설정
        if args.mode == "fusion":
            visual_feature_path = os.path.join(args.visual_root, args.visual_backbone)
            text_feature_path = os.path.join(args.text_root, "CAVAD", args.text_backbone)
        elif args.mode == "visual":
            visual_feature_path = os.path.join(args.visual_root, args.visual_backbone)
            text_feature_path = None
        else:
            visual_feature_path = None
            text_feature_path = os.path.join(args.text_root, args.mode, args.text_backbone)

        scheduler.step()
        # ✅ Validation 수행
        overall_auc, overall_ap = validate(model, args.test_list_path, args.gt_feature_path, visual_feature_path, text_feature_path)

        print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {epoch_loss:.4f}, AUC: {overall_auc:.4f}, AP: {overall_ap:.4f}")

        # ✅ 모델 저장
        if args.save_model and overall_auc > best_auc:
            best_auc = overall_auc
            save_model(model, os.path.join(result_path, "best_model.pth"))

        # ✅ Category-wise Validation
        category_list = sorted(glob.glob(os.path.join(args.category_test_list_path, "*.txt")))
        for category_file in category_list:
            category_name = os.path.splitext(os.path.basename(category_file))[0]
            category_gt_path = os.path.join(args.category_gt_feature_path, category_name+"_GT.npy")
            category_auc, category_ap = validate(model, category_file, category_gt_path, visual_feature_path, text_feature_path)
            print(f"Category: {category_name}, AUC: {category_auc:.4f}, AP: {category_ap:.4f}")
        
        print("-" * 60)

    if args.save_model:
        validate_and_plot(model)

if __name__ == "__main__":
    train()