import os
import shutil
import random

# テキストファイルが入っているディレクトリ
TXT_DIR = r"C:\Users\ryoma\ASE\anno"
# 画像ファイルが入っているディレクトリ
IMG_DIR = r"C:\Users\ryoma\ASE\img"
# 出力先
DST_DIR = r"C:\Users\ryoma\ASE\data"
TRAIN_DIR = os.path.join(DST_DIR, "train")
VAL_DIR = os.path.join(DST_DIR, "val")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

IMG_EXT = ".jpg"
TXT_EXT = ".txt"

# ペアをリストアップ
pairs = []
for fname in os.listdir(TXT_DIR):
    if fname.lower().endswith(TXT_EXT):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, base + IMG_EXT)
        txt_path = os.path.join(TXT_DIR, base + TXT_EXT)
        if os.path.exists(img_path):
            pairs.append((img_path, txt_path))

# シャッフルして8:2に分割
random.shuffle(pairs)
n_train = int(len(pairs) * 0.8)
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:]

def copy_pairs(pairs, out_dir):
    for img_path, txt_path in pairs:
        shutil.copy(img_path, os.path.join(out_dir, os.path.basename(img_path)))
        shutil.copy(txt_path, os.path.join(out_dir, os.path.basename(txt_path)))

copy_pairs(train_pairs, TRAIN_DIR)
copy_pairs(val_pairs, VAL_DIR)

print(f"train: {len(train_pairs)}ペア, val: {len(val_pairs)}ペアをコピーしました。")