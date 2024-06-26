import os
import sys
import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import process_feat

old_path = input("train_path: ").rstrip()

if os.path.exists(old_path):
    train_path = old_path
else:
    raise ValueError("Invalid path")

feat_prefix = train_path.replace("train", "train-200")
os.makedirs(feat_prefix, exist_ok=True)

max_len = 200
train_list = os.listdir(train_path)
for feat_path in tqdm.tqdm(train_list):
    feat_path = os.path.join(train_path, feat_path.strip('\n'))
    feat_path = feat_path.strip('\n')
    v_feat = np.array(np.load(feat_path), dtype=np.float32)
    v_feat = process_feat(v_feat, max_len, is_random=False)
    output_path = feat_path.replace("train", "train-200")
    np.save(output_path, v_feat)