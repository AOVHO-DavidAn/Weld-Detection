import os
import shutil
import random
import yaml
import tqdm
from pathlib import Path
import cv2
import numpy as np

def split_dataset(source_images_dir, source_labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, target_dtype='uint8', save_negative=False):
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        source_images_dir: 源图像目录
        source_labels_dir: 源标签目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        target_dtype: 目标数据类型 ('uint8', 'uint16', 'float32', 'original')
        save_negative: 是否只保存负片图像（True时只保存负片，False时保存原图）
    """
    
    # 确保比例加起来等于1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    # 验证数据类型参数
    valid_dtypes = ['uint8', 'uint16', 'float32', 'original']
    assert target_dtype in valid_dtypes, f"target_dtype 必须是 {valid_dtypes} 中的一个"
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    
    # 创建目录结构 - 根据是否保存负片决定目录名称
    if save_negative:
        dirs_to_create = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
    else:
        dirs_to_create = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
    
    for dir_name in dirs_to_create:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(Path(source_images_dir).glob(ext))
    
    # 过滤出有对应标签文件的图像
    valid_files = []
    for img_file in image_files:
        label_file = Path(source_labels_dir) / (img_file.stem + '.txt')
        if label_file.exists():
            valid_files.append(img_file.stem)
    
    print(f"找到 {len(valid_files)} 个有效的图像-标签对")
    print(f"目标数据类型: {target_dtype}")
    print(f"图像类型: {'负片' if save_negative else '原图'}")
    
    # 随机打乱文件列表
    random.shuffle(valid_files)
    
    # 计算分割点
    total_files = len(valid_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # 分割文件列表
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    
    def convert_image_dtype(img, target_dtype, original_dtype, img_name=""):
        """转换图像数据类型"""
        if target_dtype == 'original' or original_dtype == target_dtype:
            return img
        
        print(f"  转换: {original_dtype} -> {target_dtype}")
        print(f"  文件名：{img_name}")
        
        if target_dtype == 'uint8':
            if original_dtype == np.uint16:
                # uint16 -> uint8: 缩放到0-255范围
                if img.max() > 255:
                    return (img / 256).astype(np.uint8)
                else:
                    return img.astype(np.uint8)
            elif original_dtype == np.float32 or original_dtype == np.float64:
                # float -> uint8: 假设float范围是0-1，缩放到0-255
                if img.max() <= 1.0:
                    return (img * 255).astype(np.uint8)
                else:
                    # 如果float范围是0-255
                    return np.clip(img, 0, 255).astype(np.uint8)
            else:
                return img.astype(np.uint8)
        
        elif target_dtype == 'uint16':
            if original_dtype == np.uint8:
                # uint8 -> uint16: 扩展到0-65535范围
                return (img.astype(np.uint16) * 256)
            elif original_dtype == np.float32 or original_dtype == np.float64:
                # float -> uint16
                if img.max() <= 1.0:
                    return (img * 65535).astype(np.uint16)
                else:
                    return np.clip(img, 0, 65535).astype(np.uint16)
            else:
                return img.astype(np.uint16)
        
        elif target_dtype == 'float32':
            if original_dtype == np.uint8:
                # uint8 -> float32: 归一化到0-1范围
                return img.astype(np.float32) / 255.0
            elif original_dtype == np.uint16:
                # uint16 -> float32: 归一化到0-1范围
                return img.astype(np.float32) / 65535.0
            else:
                return img.astype(np.float32)
        
        return img

    def convert_to_negative(img, target_dtype):
        """将图像转换为负片"""
        if target_dtype == 'uint8':
            return 255 - img
        elif target_dtype == 'uint16':
            return 65535 - img
        elif target_dtype == 'float32':
            if img.max() <= 1.0:
                return 1.0 - img
            else:
                return img.max() - img
        elif target_dtype == 'original':
            if img.dtype == np.uint8:
                return 255 - img
            elif img.dtype == np.uint16:
                return 65535 - img
            elif img.dtype in [np.float32, np.float64]:
                if img.max() <= 1.0:
                    return 1.0 - img
                else:
                    return img.max() - img
            else:
                # 对于其他类型，归一化后反转
                img_norm = (img - img.min()) / (img.max() - img.min())
                return 1.0 - img_norm
        else:
            return img
    
    # 复制文件到对应目录
    def copy_files(file_list, subset_name):
        conversion_count = 0
        negative_count = 0
        
        for file_stem in tqdm.tqdm(file_list, desc=f"Copying {subset_name} files"):
            # 处理图像文件
            for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
                src_img = Path(source_images_dir) / f"{file_stem}.{ext}"
                if src_img.exists():
                    # 确定目标图像路径
                    dst_img = output_path / 'images' / subset_name / src_img.name
                    
                    # 读取图像
                    img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        original_dtype = img.dtype
                        
                        # 进行数据类型转换
                        if target_dtype == 'original':
                            converted_img = img
                        else:
                            converted_img = convert_image_dtype(img, target_dtype, original_dtype, img_name=str(src_img))
                            if original_dtype != converted_img.dtype:
                                conversion_count += 1
                        
                        # 决定保存原图还是负片
                        if save_negative:
                            # 转换为负片并保存
                            final_img = convert_to_negative(converted_img, target_dtype)
                            negative_count += 1
                        else:
                            # 保存原图
                            final_img = converted_img
                        
                        # 保存图像
                        if target_dtype == 'float32':
                            # float32 保存为.npy和可视化版本
                            dst_img_npy = dst_img.with_suffix('.npy')
                            np.save(dst_img_npy, final_img)
                            # 可视化版本
                            visual_img = (final_img * 255).astype(np.uint8) if final_img.max() <= 1.0 else final_img.astype(np.uint8)
                            cv2.imwrite(str(dst_img), visual_img)
                        else:
                            # uint8 或 uint16 直接保存
                            cv2.imwrite(str(dst_img), final_img)
                    break

            # 复制标签文件
            src_label = Path(source_labels_dir) / f"{file_stem}.txt"
            if src_label.exists():
                dst_label = output_path / 'labels' / subset_name / src_label.name
                shutil.copy2(src_label, dst_label)
        
        if conversion_count > 0:
            print(f"  {subset_name}: 转换了 {conversion_count} 个图像的数据类型")
        if save_negative and negative_count > 0:
            print(f"  {subset_name}: 生成了 {negative_count} 个负片图像")
        elif not save_negative:
            print(f"  {subset_name}: 保存了 {len(file_list)} 个原图")
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print("数据集划分完成！")
    return train_files, val_files, test_files

def create_data_yaml(output_dir, class_mapping_file=None, custom_classes=None, save_negative=False):
    """
    创建YOLOv8训练所需的data.yaml文件
    
    Args:
        output_dir: 输出目录
        class_mapping_file: 类别映射文件路径
        custom_classes: 自定义类别列表
        save_negative: 是否为负片数据集创建yaml文件
    """
    
    output_path = Path(output_dir)
    
    # 读取类别信息
    if class_mapping_file and os.path.exists(class_mapping_file):
        with open(class_mapping_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    elif custom_classes:
        classes = custom_classes
    else:
        # 默认类别
        classes = ['气孔', '未熔透', '未熔合', '裂纹', '夹渣', '伪缺陷', '焊缝']
    
    # 创建data.yaml内容
    data_yaml = {
        'path': str(output_path.absolute()),  # 数据集根目录
        'train': 'images/train',  # 训练图像目录（相对于path）
        'val': 'images/val',      # 验证图像目录（相对于path）
        'test': 'images/test',    # 测试图像目录（相对于path）
        'nc': len(classes),       # 类别数量
        'names': {i: name for i, name in enumerate(classes)}  # 类别名称映射
    }
    
    # 保存data.yaml文件
    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    image_type = "负片" if save_negative else "原图"
    print(f"已创建 {image_type} data.yaml 文件: {yaml_file}")
    print(f"包含 {len(classes)} 个类别: {classes}")
    
    return str(yaml_file)

def main():
    """主函数"""
    
    # 设置随机种子，确保结果可重现
    random.seed(42)
    
    # 配置路径
    source_images_dir = "datasets/Raw_data/images"
    source_labels_dir = "datasets/Raw_data_jpg/labels"
    base_output_dir = "datasets/yolo_RTA_tif"
    classes_file = "datasets/Raw_data_jpg/classes.txt"

    # 检查源目录是否存在
    if not os.path.exists(source_images_dir):
        print(f"错误: 源图像目录不存在: {source_images_dir}")
        return
    
    if not os.path.exists(source_labels_dir):
        print(f"错误: 源标签目录不存在: {source_labels_dir}")
        return
    
    # 数据类型选择
    print("选择目标数据类型:")
    print("1. uint8   - 8位无符号整数 (0-255)")
    print("2. uint16  - 16位无符号整数 (0-65535)")
    print("3. float32 - 32位浮点数 (0.0-1.0)")
    print("4. original - 保持原始格式")
    
    dtype_choice = input("请选择 (1/2/3/4) [默认:1]: ").strip()
    
    dtype_map = {
        '1': 'uint8',
        '2': 'uint16', 
        '3': 'float32',
        '4': 'original',
        '': 'uint8'  # 默认选择
    }
    
    target_dtype = dtype_map.get(dtype_choice, 'uint8')
    
    # 图像类型选择
    print("\n选择图像类型:")
    print("1. 原图 - 保存原始图像")
    print("2. 负片 - 保存负片图像")
    
    image_choice = input("请选择 (1/2) [默认:1]: ").strip()
    save_negative = image_choice == '2'
    
    print(f"\n已选择数据类型: {target_dtype}")
    print(f"图像类型: {'负片' if save_negative else '原图'}")
    
    # 根据选择确定输出目录名称
    dtype_suffix = f"_{target_dtype}" if target_dtype != 'uint8' else ""
    image_suffix = "_negative" if save_negative else ""
    output_dir = f"{base_output_dir}{dtype_suffix}{image_suffix}"
    
    print("开始数据集划分...")
    print(f"源图像目录: {source_images_dir}")
    print(f"源标签目录: {source_labels_dir}")
    print(f"输出目录: {output_dir}")
    
    # 执行数据集划分
    train_files, val_files, test_files = split_dataset(
        source_images_dir=source_images_dir,
        source_labels_dir=source_labels_dir,
        output_dir=output_dir,
        train_ratio=0.7,  # 70% 训练集
        val_ratio=0.2,    # 20% 验证集
        test_ratio=0.1,   # 10% 测试集
        target_dtype=target_dtype,
        save_negative=save_negative
    )
    
    # 创建data.yaml文件
    print("\n创建data.yaml文件...")
    yaml_file = create_data_yaml(
        output_dir=output_dir,
        class_mapping_file=classes_file,
        save_negative=save_negative
    )
    
    print(f"\n数据集划分完成！")
    print(f"输出目录: {output_dir}")
    print(f"数据类型: {target_dtype}")
    print(f"图像类型: {'负片' if save_negative else '原图'}")
    print(f"YOLOv8配置文件: {yaml_file}")
    
    # 显示目录结构
    print(f"\n生成的目录结构:")
    print(f"{output_dir}/")
    print(f"├── images/")
    print(f"│   ├── train/  ({len(train_files)} files)")
    print(f"│   ├── val/    ({len(val_files)} files)")
    print(f"│   └── test/   ({len(test_files)} files)")
    print(f"├── labels/")
    print(f"│   ├── train/  ({len(train_files)} files)")
    print(f"│   ├── val/    ({len(val_files)} files)")
    print(f"│   └── test/   ({len(test_files)} files)")
    print(f"└── data.yaml")

if __name__ == "__main__":
    main()
