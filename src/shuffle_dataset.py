import os
import random
from glob import glob
from tqdm import tqdm

DATASET_PATH = "../dataset"
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

def is_image_file(filename):
    return filename.lower().endswith(EXTENSIONS)

def shuffle_and_rename_images():
    for class_dir in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_dir)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if is_image_file(f)]
        random.shuffle(images)

        print(f"Shuffling {class_dir} ({len(images)} images)")

        for idx, filename in enumerate(tqdm(images, desc=f"  Renaming in {class_dir}")):
            old_path = os.path.join(class_path, filename)
            ext = os.path.splitext(filename)[-1]
            new_filename = f"{class_dir}_{idx:04d}{ext}"
            new_path = os.path.join(class_path, new_filename)
            os.rename(old_path, new_path)

if __name__ == "__main__":
    shuffle_and_rename_images()
    print("All class folders shuffled and renamed.")
