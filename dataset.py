import torch
import numpy as np
import os
from torch.utils.data import Dataset
from config import args
import torch.nn.functional as F

class VADDataset(Dataset):
    def __init__(self, is_anomaly=True):
        self.is_anomaly = is_anomaly
        self.base_path = args.base_path
        self.visual_root = os.path.join(args.visual_root, args.visual_backbone)
        self.text_root = os.path.join(args.text_root, args.text_backbone)
        self.data_list, self.categories = self._load_feature_list(is_anomaly)
        self.num_segments_per_video = args.num_segments_per_video
        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(self.categories))}
        self.num_categories = len(self.category_to_idx)
        print(f"Loaded {len(self.data_list)} samples (Anomaly: {self.is_anomaly})")

    def _load_feature_list(self, is_anomaly):
        data_list = []
        categories = set()
        with open(args.train_list, "r") as f:
            for line in f:
                video_name = line.strip()
                category = os.path.dirname(video_name)
                if category == "Training_Normal_Videos_Anomaly":
                    is_anomaly_video = False
                else:
                    is_anomaly_video = True
                    categories.add(category)
                if is_anomaly != is_anomaly_video:
                    continue
                video_base = os.path.basename(video_name).replace("x264.mp4", "")
                visual_category_path = os.path.join(self.visual_root, category)
                visual_feature_files = [f for f in os.listdir(visual_category_path) if f.startswith(video_base) and f.endswith(".npy")]
                if len(visual_feature_files) == 0:
                    continue
                visual_feature_path = os.path.join(visual_category_path, visual_feature_files[0])
                text_category_path = os.path.join(self.text_root, category)
                text_feature_files = [f for f in os.listdir(text_category_path) if f.startswith(video_base) and f.endswith(".npy")]
                if len(text_feature_files) == 0:
                    continue
                text_feature_path = os.path.join(text_category_path, text_feature_files[0])
                data_list.append((visual_feature_path, text_feature_path, is_anomaly_video, category))
        return data_list, categories

    def _process_feature(self, feature):
        num_segments, feature_dim = feature.shape
        target_segments = self.num_segments_per_video
        new_feat = np.zeros((target_segments, feature_dim), dtype=np.float32)
        split_idx = np.linspace(0, num_segments, target_segments + 1, dtype=int)
        for i in range(target_segments):
            start, end = split_idx[i], split_idx[i+1]
            if start != end:
                new_feat[i] = np.mean(feature[start:end], axis=0)
            else:
                new_feat[i] = feature[min(start, num_segments - 1)]
        return new_feat

    def _z_score_normalize(self, feature):
        mean = feature.mean(axis=0, keepdims=True)
        std = feature.std(axis=0, keepdims=True) + 1e-6
        return (feature - mean) / std

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        visual_path, text_path, label, category = self.data_list[idx]
        visual_feature = np.load(visual_path, allow_pickle=False)
        text_feature = np.load(text_path, allow_pickle=False)
        visual_feature = self._process_feature(visual_feature)
        text_feature = self._process_feature(text_feature)
        visual_tensor = torch.tensor(visual_feature, dtype=torch.float32)
        text_tensor = torch.tensor(text_feature, dtype=torch.float32)
        feature = torch.cat((visual_tensor, text_tensor), dim=1)
        if category == "Training_Normal_Videos_Anomaly":
            cat_onehot = torch.zeros(self.num_categories, dtype=torch.float32)
        else:
            cat_idx = self.category_to_idx[category]
            cat_onehot = F.one_hot(torch.tensor(cat_idx), num_classes=self.num_categories).float()
        return feature, torch.tensor([label], dtype=torch.float32), cat_onehot
