#!/usr/bin/bash

#SBATCH -J CAVAD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


# MODE: "CAVAD", "SAVAD", "visual", "fusion"
MODE="fusion"

# FUSION_TYPE: "linear", "concat", "cross_attention"
FUSION_TYPE="linear"

# VISUAL_BACKBONE: "ViT16B", "ViT32B", "ViT14L"
VISUAL_BACKBONE="ViT16B"

# TEXT_BACKBONE: "ViT16B", "ViT32B", "ViT14L"
TEXT_BACKBONE="ViT16B"

# RESULT_FOLDER: 
RESULT_FOLDER="result1"

# SAVE_MODEL: 저장하려면 "--save_model", 저장하지 않으려면 빈 문자열 ""
SAVE_MODEL="--save_model"



# ✅ 학습 실행
python train.py --mode $MODE \
                         --fusion-type $FUSION_TYPE \
                         --visual_backbone $VISUAL_BACKBONE \
                         --text_backbone $TEXT_BACKBONE \
                         --result_folder $RESULT_FOLDER \
                         $SAVE_MODEL
