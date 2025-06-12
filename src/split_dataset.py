import os
import shutil
import random


ORIGIN_DIR = "../dataset_resized"
TARGET_DIR = "../dataset"
CLASS_NAMES = ["sorento", "santafe", "grandeur", "k5"]
train_count = 190
seed = 42

random.seed(seed)

for cls in CLASS_NAMES:
    src_dir = os.path.join(ORIGIN_DIR, cls)
    if not os.path.isdir(src_dir):
        print(f"[WARN] Can not find Directory: {src_dir}")
        continue

    images = sorted(os.listdir(src_dir))
    random.shuffle(images)

    train_images = images[:train_count]
    test_images  = images[train_count:]

    for subset, subset_images in [("train", train_images), ("test", test_images)]:
        dest_dir = os.path.join(TARGET_DIR, subset, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for img_name in subset_images:
            src = os.path.join(src_dir, img_name)
            dst = os.path.join(dest_dir, img_name)
            shutil.copy2(src, dst)

        print(f"{cls} → {subset} ({len(subset_images)})")

print("\nsplit is COMPLETE: Train(180/CLASS), Test(20/CLASS)")
