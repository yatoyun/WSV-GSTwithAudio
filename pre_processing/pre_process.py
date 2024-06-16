import os
import tqdm
import numpy as np
from utils import process_feat

dataset_name = "xd"
version = input("version: ")
if version not in ["clip", "i3d", "audio"]:
    raise ValueError("Invalid version")

if version == "clip":
    train_path = f"../list/{dataset_name}/train-clip.list"

elif version == "audio":
    train_path = "/home/yukaneko/dev/CLIP-TSA_dataset/xd/features/vggish-features/train/"

elif version == "i3d":
    train_path = f"../list/{dataset_name}/train.list"


feat_prefix = f"../data/{dataset_name}-i3d"

if version != "audio":
    train_list = list(open(train_path))


print(feat_prefix)
if version == "i3d":
# make dir
    os.makedirs(f"../data/{dataset_name}-i3d/train-200", exist_ok=True)

    max_len = 200
    print(len(train_list))
    for path in tqdm.tqdm(train_list):
        if dataset_name == "sh":
            path = path.split(" ")[0]
        feat_path = os.path.join(feat_prefix, path.strip('\n'))
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        v_feat = process_feat(v_feat, max_len, is_random=False)
        output_path = feat_path.replace("train", "train-200")
        np.save(output_path, v_feat)

elif version == "audio":
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

else:
    feat_prefix = train_list[0].split("train")[0]+"train-200/"
    os.makedirs(feat_prefix, exist_ok=True)

    max_len = 200
    print(len(train_list))
    for feat_path in tqdm.tqdm(train_list):
        # feat_path = os.path.join(feat_prefix, path.strip('\n'))
        feat_path = feat_path.strip('\n')
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        v_feat = process_feat(v_feat, max_len, is_random=False)
        output_path = feat_path.replace("train", "train-200")
        
        video_class_name = feat_path.split("/")[-2]
        os.makedirs(feat_prefix+video_class_name, exist_ok=True)
        np.save(output_path, v_feat)