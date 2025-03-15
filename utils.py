import torch 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import re

def save_model(model, path):
    """📌 모델 저장 함수"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def extract_category_from_filename(filename):
    """ 📌 파일명에서 첫 숫자 전까지의 부분을 카테고리로 추출 """
    match = re.match(r"([^\d]+)", filename)
    return match.group(1) if match else "Unknown"

