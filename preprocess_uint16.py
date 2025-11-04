import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# --- 配置 ---
# 包含 uint16 .tif 图像的数据集根目录
INPUT_DATASET_DIR = Path('datasets/yolo_GuanWang_dataset')

# 转换后数据集的输出目录
OUTPUT_DATASET_DIR = Path('datasets/yolo_GuanWang_dataset_uint8')
# ---

def convert_uint16_to_uint8(input_dir: Path, output_dir: Path):
    """
    将数据集中的 uint16 .tif 图像转换为 uint8 .png 图像，并复制标签文件。
    """
    if output_dir.exists():
        print(f"警告: 输出目录 '{output_dir}' 已存在。其中的内容可能会被覆盖。")
    
    # 遍历 train, val, test 子目录
    for split in ['train', 'val', 'test']:
        src_img_dir = input_dir / 'images' / split
        src_lbl_dir = input_dir / 'labels' / split
        
        dst_img_dir = output_dir / 'images' / split
        dst_lbl_dir = output_dir / 'labels' / split
        
        if not src_img_dir.exists():
            print(f"跳过: 找不到源目录 '{src_img_dir}'")
            continue

        # 创建输出目录
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n正在处理 '{split}' 集...")
        
        image_files = list(src_img_dir.glob('*.tif'))
        
        for img_path in tqdm(image_files, desc=f"转换 {split} 图像"):
            # 1. 读取 uint16 图像
            img_uint16 = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_uint16 is None:
                print(f"警告: 无法读取图像 {img_path}，已跳过。")
                continue

            # 2. 归一化到 [0, 1] 范围
            # 使用 np.max(img_uint16) 作为分母，以保留图像的动态范围
            # 如果图像本身的最大值小于65535，这可以防止图像整体变暗
            max_val = np.max(img_uint16)
            if max_val == 0:
                img_float = img_uint16.astype(np.float32)
            else:
                img_float = img_uint16.astype(np.float32) / max_val

            # 3. 缩放到 [0, 255] 范围并转换为 uint8
            img_uint8 = (img_float * 255.0).astype(np.uint8)

            # 4. 保存为 .tif 格式 (无损)
            base_name = img_path.stem
            output_img_path = dst_img_dir / f"{base_name}.tif"
            cv2.imwrite(str(output_img_path), img_uint8)

        # 5. 复制对应的标签文件
        if src_lbl_dir.exists():
            print(f"正在复制 '{split}' 集的标签...")
            shutil.copytree(src_lbl_dir, dst_lbl_dir, dirs_exist_ok=True)

    # 6. 复制 data.yaml 文件
    src_yaml = input_dir / 'data.yaml'
    if src_yaml.exists():
        shutil.copy2(src_yaml, output_dir / 'data.yaml')
        print(f"\n已复制 'data.yaml' 文件到 '{output_dir}'")

    print(f"\n处理完成！新的 uint8 数据集已保存在: '{output_dir}'")


if __name__ == '__main__':
    convert_uint16_to_uint8(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR)