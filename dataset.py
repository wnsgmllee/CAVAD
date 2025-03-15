import torch
import numpy as np
import os
from torch.utils.data import Dataset
from config import args


class VADDataset(Dataset):
    def __init__(self, mode, is_anomaly=True):
        """
        📌 데이터셋 로드 (Single & Fusion 지원)
        mode: "CAVAD", "SAVAD", "visual", "fusion"
        """
        self.mode = mode
        self.is_anomaly = is_anomaly
        self.base_path = args.base_path
        self.visual_backbone = args.visual_backbone
        self.text_backbone = args.text_backbone
        

        # ✅ Feature 경로 설정
        self.visual_root = args.visual_root
        self.text_root = args.text_root

        if self.mode == "CAVAD":
            self.text_root = os.path.join(self.text_root, "CAVAD", self.text_backbone)
        elif self.mode == "SAVAD":
            self.text_root = os.path.join(self.text_root, "SAVAD", self.text_backbone)
        elif self.mode == "visual":
            self.visual_root = os.path.join(self.visual_root, self.visual_backbone)
        elif self.mode == "fusion":
            self.text_root = os.path.join(self.text_root, "CAVAD", self.text_backbone)
            self.visual_root = os.path.join(self.visual_root, self.visual_backbone)

        self.data_list = self._load_feature_list(is_anomaly)
        self.num_segments_per_video = args.num_segments_per_video

        print(f"📌 Loaded {len(self.data_list)} samples (Mode: {self.mode}, Anomaly: {self.is_anomaly})")

    def _load_feature_list(self, is_anomaly):
        """ 📌 Feature 파일 리스트 로드 """
        data_list = []
        with open(args.train_list, "r") as f:
            for line in f:
                video_name = line.strip()
                category = os.path.dirname(video_name)
                is_anomaly_video = category != "Training_Normal_Videos_Anomaly"
                if is_anomaly != is_anomaly_video:
                    continue

                video_base = os.path.basename(video_name).replace("x264.mp4", "")

                if self.mode == "fusion":
                    category_visual_path = os.path.join(self.visual_root, category)
                    category_text_path = os.path.join(self.text_root, category)
                    visual_files = [f for f in os.listdir(category_visual_path) if f.startswith(video_base) and f.endswith(".npy")]
                    visual_feature = os.path.join(category_visual_path, visual_files[0])
                    text_files = [f for f in os.listdir(category_text_path) if f.startswith(video_base) and f.endswith(".npy")]
                    text_feature = os.path.join(category_text_path, text_files[0])
                    data_list.append((visual_feature, text_feature, is_anomaly_video))

                else:
                    category_feature_path = os.path.join(self.visual_root if "visual" == self.mode else self.text_root, category)
                    feature_files = [f for f in os.listdir(category_feature_path) if f.startswith(video_base) and f.endswith(".npy")]
                    feature = os.path.join(category_feature_path, feature_files[0])
                    data_list.append((feature, is_anomaly_video))


        return data_list

    def _process_feature(self, feature):
        num_segments, feature_dim = feature.shape
        target_segments = self.num_segments_per_video

        if num_segments == target_segments:
            return feature
        elif num_segments < target_segments:
            pad_size = target_segments - num_segments
            pad = np.zeros((pad_size, feature_dim), dtype=np.float32)
            return np.concatenate((feature, pad), axis=0)
        else:
            indices = np.linspace(0, num_segments - 1, target_segments).astype(int)
            return feature[indices]

    def _z_score_normalize(self, feature):
        mean = feature.mean(axis=0, keepdims=True)
        std = feature.std(axis=0, keepdims=True) + 1e-6  # Avoid division by zero
        return (feature - mean) / std

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """ 📌 데이터 로드 및 전처리 """
        if self.mode == "fusion":
            visual_path, text_path, label = self.data_list[idx]
            visual_feature = np.load(visual_path, allow_pickle=False)
            text_feature = np.load(text_path, allow_pickle=False)

            visual_feature = self._process_feature(visual_feature)
            text_feature = self._process_feature(text_feature)

            visual_feature = self._z_score_normalize(visual_feature)
            text_feature = self._z_score_normalize(text_feature)

            visual_feature = torch.tensor(visual_feature, dtype=torch.float32)
            text_feature = torch.tensor(text_feature, dtype=torch.float32)

            return (visual_feature, text_feature), torch.tensor([label], dtype=torch.float32)
        else:
            feature_path, label = self.data_list[idx]
            feature = np.load(feature_path, allow_pickle=False)

            feature = self._process_feature(feature)

            feature = self._z_score_normalize(feature)

            feature = torch.tensor(feature, dtype=torch.float32)

            return (feature), torch.tensor([label], dtype=torch.float32)