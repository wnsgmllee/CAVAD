import os 
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from config import args
from natsort import natsorted

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


def validate(model, test_list_path, gt_path, visual_feature_path, text_feature_path):
    """ 📌 모델 검증 """
    model.eval()
    test_videos = np.loadtxt(test_list_path, dtype=str)
    test_videos = natsorted(test_videos)
    # print(f"📌 test_list_path: {test_list_path}")

    pred_scores = compute_model_scores(model, test_videos, visual_feature_path, text_feature_path)
    gt_labels = np.load(gt_path, allow_pickle=False)

    roc_auc = roc_auc_score(gt_labels, pred_scores) if len(pred_scores) > 0 else 0
    ap_score = average_precision_score(gt_labels, pred_scores) 

    # print(f"Validate Results: AUC: {roc_auc:.4f} / AP: {ap_score:.4f}")
    return roc_auc, ap_score


##############################################################################################
def get_video_frame_counts(visual_feature_root, test_videos): # 정상
    """
    📌 각 테스트 비디오의 프레임 개수 계산
    - 정렬된 영상명 리스트 사용
    """
    frame_counts = {}
    for video_name in sorted(test_videos, key=lambda x: x.lower()):
        video_name = video_name.replace("x264", "")
        for category in sorted(os.listdir(visual_feature_root)):  # ✅ 카테고리도 정렬
            category_path = os.path.join(visual_feature_root, category)
            if not os.path.isdir(category_path):
                continue

            expected_filename = f"{video_name}16frames.npy"
            visual_files = [f for f in os.listdir(category_path) if f == expected_filename]

            if visual_files:
                visual_feature_path = os.path.join(category_path, visual_files[0])
                visual_feature = np.load(visual_feature_path, allow_pickle=False)
                frame_counts[video_name] = visual_feature.shape[0] * 16  # segment 개수 * 16
                break

    return frame_counts

def plot_anomaly_scores(gt_labels, pred_scores, test_videos, frame_counts, save_path):
    """
    📌 GT와 예측된 anomaly score를 시각화하여 저장하는 함수
    - 정렬된 영상명 리스트 사용
    """
    save_path = os.path.join(save_path, "vis_results")
    os.makedirs(save_path, exist_ok=True)
    start_idx = 0

    # ✅ test_videos 정렬
    for video_name in sorted(test_videos, key=lambda x: x.lower()):
        video_name = video_name.replace("x264","")
        if video_name not in frame_counts:
            continue
        frame_count = frame_counts[video_name]
        end_idx = start_idx + frame_count

        plt.figure(figsize=(12, 5))
        plt.plot(range(frame_count), gt_labels[start_idx:end_idx], label="Ground Truth", linestyle='dashed', alpha=0.7)
        plt.plot(range(frame_count), pred_scores[start_idx:end_idx], label="Predicted Score", alpha=0.9)
        plt.ylim(0, 1)
        plt.xlabel("Frame")
        plt.ylabel("Anomaly Score")
        plt.title(f"Anomaly Score Plot: {video_name}")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"{video_name}.png"))
        plt.close()

        start_idx = end_idx

def compute_model_scores_frame_level(visual_feature_root, text_feature_root, model, test_videos, device="cuda"):
    """
    📌 모델 예측 점수 로드 및 Frame-Level 변환 (Visual + Text Feature 사용)
    """
    all_pred_frames = []

    for video_name in test_videos:
        video_name = video_name.replace("x264", "")

        found = False  # 🔥 파일을 찾았는지 여부 추적

        for category in sorted(os.listdir(visual_feature_root)):  # ✅ 카테고리 정렬
            visual_category_path = os.path.join(visual_feature_root, category)
            text_category_path = os.path.join(text_feature_root, category)  # ✅ 추가

            if not os.path.isdir(visual_category_path) or not os.path.isdir(text_category_path):
                continue  # 폴더가 아니면 스킵

            expected_filename = f"{video_name}16frames.npy"
            visual_files = [f for f in os.listdir(visual_category_path) if f == expected_filename]
            text_files = [f for f in os.listdir(text_category_path) if f == expected_filename]  # ✅ 추가

            if visual_files and text_files:  # ✅ 두 개의 feature가 모두 존재하는 경우
                visual_feature_path = os.path.join(visual_category_path, visual_files[0])
                text_feature_path = os.path.join(text_category_path, text_files[0])

                visual_feature = torch.tensor(np.load(visual_feature_path), dtype=torch.float32).to(device)
                text_feature = torch.tensor(np.load(text_feature_path), dtype=torch.float32).to(device)  # ✅ 추가

                with torch.no_grad():
                    scores = model(visual_feature, text_feature).cpu().numpy()  # ✅ 수정: text_feature 추가

                scores = 1 / (1 + np.exp(-scores))
                frame_scores = np.repeat(scores, 16, axis=0)
                all_pred_frames.append(frame_scores)

                found = True  # ✅ feature를 찾았음 표시
                break  # ✅ 정확한 feature를 찾았으니 종료

        if not found:
            print(f"⚠️ Warning: Feature not found for {video_name}")

    return np.concatenate(all_pred_frames) if all_pred_frames else np.array([])


def validate_and_plot(model):
    """
    📌 Best epoch의 모델을 로드하여 예측값을 계산하고, GT와 함께 시각화
    """
    result_path = os.path.join("results", args.result_folder)
    os.makedirs(result_path, exist_ok=True)
    model.load_state_dict(torch.load(os.path.join(result_path, "best_model.pth")))
    model.eval()

    test_videos = load_test_list(args.test_list_path)
    frame_counts = get_video_frame_counts(os.path.join(args.visual_root, args.visual_backbone), test_videos)

    gt_labels = np.load(args.gt_feature_path, allow_pickle=False)
    pred_scores = compute_model_scores_frame_level(
        os.path.join(args.visual_root, args.visual_backbone),
        os.path.join(args.text_root, "CAVAD", args.text_backbone),  # ✅ text feature 경로 추가
        model,
        test_videos
    )

    if len(gt_labels) != len(pred_scores):
        raise ValueError(f"❌ GT와 예측값 길이가 다름! GT: {len(gt_labels)}, 예측: {len(pred_scores)}")

    plot_anomaly_scores(gt_labels, pred_scores, test_videos, frame_counts, result_path)


##############################################################################################