import os
import json
import tqdm
import numpy as np
from PIL import Image
# from src.image_processor import ImageProcessor
# from src.json_processor import JsonProcessor
from src.utils.config import input_directory as INPUT_DIR
from src.utils.config import output_directory as OUTPUT_DIR

def normalize_image(img_array):
    """
    将图像像素值标准化到0-255范围
    """
    # 获取最小值和最大值
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    # 避免除零错误
    if max_val == min_val:
        return np.zeros_like(img_array, dtype=np.uint8)
    
    # 标准化到0-255范围
    normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def compress_images():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process TIFF images in the specified input directory
    count = 0
    print(f"Starting compression of images from {INPUT_DIR} to {OUTPUT_DIR}")
    for root, dirs, files in os.walk(INPUT_DIR):
        print(f"Processing directory: {root}")
        print(f"Found {len(files)} files in {root}")
        for file in tqdm.tqdm(files, desc=f"Processing files in {os.path.basename(root)}", total=len(files)):
            if file.endswith('.tif') or file.endswith('.tiff'):
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(OUTPUT_DIR, os.path.relpath(input_image_path, INPUT_DIR)).replace('.tif', '.jpg').replace('.tiff', '.jpg')
                
                # Create the output directory for the image (without the filename)
                output_dir_path = os.path.dirname(output_image_path)
                os.makedirs(output_dir_path, exist_ok=True)

                try:
                    # 使用PIL打开图像
                    with Image.open(input_image_path) as img:
                        # 打印图像信息用于调试
                        if count == 0:  # 只打印第一张图像的信息
                            print(f"Image mode: {img.mode}, Size: {img.size}")
                            if hasattr(img, 'tag'):
                                print(f"Image format info: {img.format}")
                        
                        # 转换为numpy数组进行处理
                        img_array = np.array(img)
                        
                        # 如果是单通道图像，需要标准化
                        if len(img_array.shape) == 2 or img.mode in ['L', 'I', 'F']:
                            # 标准化像素值到0-255范围
                            img_array = normalize_image(img_array)
                            # 转换回PIL图像
                            img_normalized = Image.fromarray(img_array, mode='L')
                            # 转换为RGB模式
                            img_rgb = img_normalized.convert('RGB')
                        elif img.mode == 'RGB':
                            # 如果已经是RGB模式，检查是否需要标准化
                            if img_array.max() > 255 or img_array.dtype != np.uint8:
                                # 对每个通道分别标准化
                                img_array_norm = np.zeros_like(img_array, dtype=np.uint8)
                                for i in range(img_array.shape[2]):
                                    img_array_norm[:,:,i] = normalize_image(img_array[:,:,i])
                                img_rgb = Image.fromarray(img_array_norm, mode='RGB')
                            else:
                                img_rgb = img
                        else:
                            # 其他模式，先转换为RGB
                            img_rgb = img.convert('RGB')
                            # 检查是否需要标准化
                            img_array = np.array(img_rgb)
                            if img_array.max() > 255 or img_array.dtype != np.uint8:
                                img_array_norm = np.zeros_like(img_array, dtype=np.uint8)
                                for i in range(img_array.shape[2]):
                                    img_array_norm[:,:,i] = normalize_image(img_array[:,:,i])
                                img_rgb = Image.fromarray(img_array_norm, mode='RGB')
                        
                        # # 在处理前后检查尺寸
                        # original_size = img.size
                        # print(f"Original size: {original_size}")
                        
                        # 保存为JPEG
                        img_rgb.save(output_image_path, 'JPEG', quality=85)
                        
                        # # 验证保存后的图像尺寸
                        # with Image.open(output_image_path) as saved_img:
                        #     saved_size = saved_img.size
                        #     print(f"Saved size: {saved_size}")
                        #     if original_size == saved_size:
                        #         print("✓ 尺寸保持不变")
                        #     else:
                        #         print("✗ 尺寸发生了变化")
                        
                except Exception as e:
                    print(f"Error processing {input_image_path}: {str(e)}")
                    continue
                
                count += 1
                if count % 10 == 0:
                    print(f"Compressed {count} images. Latest: {input_image_path}")

    print(f"Compression completed! Total images processed: {count}")

if __name__ == "__main__":
    compress_images()