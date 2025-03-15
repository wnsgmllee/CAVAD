import argparse 
import os


parser = argparse.ArgumentParser(description="UCF Crime Training")


# 🔥 Mode 설정
parser.add_argument("--mode", type=str, default="CAVAD", help="Training mode: CAVAD, SAVAD, visual, fusion")


# 📁 데이터 경로
base_path = "/data/datasets/VAD_data/UCF_Crime" 
parser.add_argument("--base_path", default="/data/datasets/VAD_data/UCF_Crime", type=str)
parser.add_argument("--visual_root", default=os.path.join(base_path, "visual"))       
parser.add_argument("--text_root", default=os.path.join(base_path, "textual"))
parser.add_argument("--train_list", default=os.path.join(base_path, "list/Anomaly_Train.txt"), type=str)
parser.add_argument('--test-list-path', default=os.path.join(base_path, "list/Temporal_Anomaly_Annotation_fix.txt"), type=str)
parser.add_argument('--gt-feature-path', default=os.path.join(base_path, "list/TestGT_padding.npy"), type=str)


# 🔢 Hyperparameter 설정
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--max_epoch", type=int, default=30, help="Max training epochs")
parser.add_argument("--lambda_contrastive", type=float, default=0.07, help="Contrastive loss weight")
parser.add_argument('--fusion-type', type=str, default='concat', choices=['concat', 'cross_attention', 'linear',])
parser.add_argument('--num-segments-per-video', default=256, type=int)
parser.add_argument('--proj-embedding', default=512, type=int)


# 🔬 Backbones
parser.add_argument("--visual_backbone", type=str, default="ViT16B", help="Visual feature backbone")
parser.add_argument("--text_backbone", type=str, default="ViT16B", help="Text feature backbone")


# 🏆 모델 저장 옵션
parser.add_argument("--save_model", action="store_true", help="Save the trained model")
parser.add_argument("--result_folder", type=str, default="result1", help="Folder name for saving results")



args = parser.parse_args()