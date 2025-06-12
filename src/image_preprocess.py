# Image preprocessing for Deep Learning Term Project
# Crop & Resize

import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 설정
INPUT_DIR = "../dataset"
OUTPUT_DIR = "../resized_dataset"
TARGET_SIZE = (300, 200)  # (width, height)

# 디렉토리 구조 복사
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in os.listdir(INPUT_DIR):
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def resize_and_pad_image(args):
    img_path, save_path = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"이미지 읽기 실패: {img_path}"

        h, w = img.shape[:2]
        scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Padding 계산
        top = (TARGET_SIZE[1] - new_h) // 2
        bottom = TARGET_SIZE[1] - new_h - top
        left = (TARGET_SIZE[0] - new_w) // 2
        right = TARGET_SIZE[0] - new_w - left

        # Padding 적용
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        cv2.imwrite(save_path, padded)
        return f"완료: {os.path.basename(img_path)}"
    except Exception as e:
        return f"예외 발생: {img_path} -> {e}"

def collect_image_paths():
    tasks = []
    for cls in os.listdir(INPUT_DIR):
        cls_path = os.path.join(INPUT_DIR, cls)
        save_path = os.path.join(OUTPUT_DIR, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            out_path = os.path.join(save_path, img_name)
            tasks.append((img_path, out_path))
    return tasks

if __name__ == "__main__":
    all_tasks = collect_image_paths()

    print(f"[INFO] 총 이미지 수: {len(all_tasks)}")
    print(f"[INFO] {cpu_count()}개의 CPU로 병렬 처리 시작...")

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap(resize_and_pad_image, all_tasks), total=len(all_tasks)):
            print(result)
