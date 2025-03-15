import os 
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from config import args

def load_test_list(file_path):
    """ 📌 test_list.txt에서 테스트할 비디오 파일 목록을 불러옴 """
    with open(file_path, "r") as f:
        return sorted([line.strip() for line in f.readlines()], key=lambda x: x.lower())

def find_feature_files(video_name, visual_feature_root, text_feature_root):
    """ 📌 특정 비디오의 Feature 파일을 찾고 경로 반환 """
    
    # ✅ 처음 등장하는 숫자 앞까지 가져오기
    match = re.split(r'(\d)', video_name, maxsplit=1)
    category = match[0] if match else video_name  # 숫자 앞 부분이 카테고리명
    if category == "Normal_Videos_":
        category = "Testing_Normal_Videos_Anomaly"

    # ✅ "x264" 제거 & "_16frames.npy" 추가
    clean_video_name = video_name.replace("x264", "").strip()
    filename = f"{clean_video_name}16frames.npy"
    
    visual_path = os.path.join(visual_feature_root, category, filename) if visual_feature_root else None
    text_path = os.path.join(text_feature_root, category, filename) if text_feature_root else None

    return visual_path, text_path

def compute_model_scores(model, test_videos, visual_feature_root, text_feature_root, device="cuda"):
    """ 📌 모델을 이용해 Frame-Level 예측 점수 계산 """
    all_pred_frames = []
    
    for video_name in test_videos:
        visual_path, text_path = find_feature_files(video_name, visual_feature_root, text_feature_root)
        if args.mode == "fusion" and (not visual_path or not text_path):
            print(f"⚠️ Warning: Missing features for {video_name}")
            continue
        if args.mode == "visual" and not visual_path:
            print(f"⚠️ Warning: Missing visual feature for {video_name}")
            continue
        if args.mode != "fusion" and args.mode != "visual" and not text_path:
            print(f"⚠️ Warning: Missing text feature for {video_name}")
            continue

        # ✅ Feature 로드
        visual_feature = torch.tensor(np.load(visual_path), dtype=torch.float32).to(device) if visual_path else None
        text_feature = torch.tensor(np.load(text_path), dtype=torch.float32).to(device) if text_path else None

        # ✅ 모델 예측
        with torch.no_grad():
            if args.mode == "fusion":
                scores = model(visual_feature, text_feature).cpu().numpy()
            else:
                scores = model(visual_feature if args.mode == "visual" else text_feature).cpu().numpy()

        scores = 1 / (1 + np.exp(-scores))
        frame_scores = np.repeat(scores, 16, axis=0)
        all_pred_frames.append(frame_scores)

    return np.concatenate(all_pred_frames) if all_pred_frames else np.array([])



def validate(model, visual_feature_path, text_feature_path):
    """ 📌 모델 검증 """
    model.eval()
    test_videos = np.loadtxt(args.test_list_path, dtype=str)
    pred_scores = compute_model_scores(model, test_videos, visual_feature_path, text_feature_path)

    gt_labels = np.load(args.gt_feature_path, allow_pickle=False)
    roc_auc = roc_auc_score(gt_labels, pred_scores) if len(pred_scores) > 0 else 0

    print(f"Validate Results: AUC: {roc_auc:.4f}")
    return roc_auc