from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os
import yaml

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 或者使用系统字体
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统中的中文字体
        font_paths = [
            '/home/yukun/.fonts/SourceHanSansSC-Regular.otf',  # 自定义下载的字体路径
            '/home/yukun/.fonts/SimHei.ttf',  # 自定义下载的字体路径
        ]
        
        for font_path in font_paths:
            print(f"检查字体路径: {font_path}")
            if os.path.exists(font_path): 
                print(f"找到字体: {font_path}")
                prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = prop.get_name()
                print(f"使用字体: {font_path}")
                return prop
        
        # 如果没有找到合适的字体，使用默认设置
        print("未找到中文字体，使用默认设置")
        return None
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        return None

# 初始化字体
chinese_font = setup_chinese_font()

def preprocess_image(image_path):
    """将单通道图像转换为3通道RGB图像"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 检查图像通道数
    if len(img.shape) == 2:  # 灰度图像
        # 将灰度图像转换为3通道RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 1:  # 单通道图像
        # 复制通道创建3通道图像
        img_rgb = np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # 已经是3通道
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"不支持的图像格式，通道数: {img.shape}")
    
    return img_rgb

def load_class_names(yaml_file):
    """加载类别名称"""
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])

    else:
        # 默认类别名称
        return ['气孔', '未熔透', '未熔合', '裂纹', '夹渣', '伪缺陷', '焊缝']

def read_yolo_labels(label_path, img_width, img_height):
    """读取YOLO格式的标签文件"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # 转换为左上角坐标
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'center': [x_center, y_center],
                        'size': [width, height]
                    })
    return labels

def visualize_ground_truth(image_path, label_path, class_names, save_path=None):
    """可视化真实标签"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # 读取标签
    labels = read_yolo_labels(label_path, img_width, img_height)
    
    # 创建matplotlib图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    # 绘制每个标签
    for i, label in enumerate(labels):
        bbox = label['bbox']
        class_id = label['class_id']
        
        # 获取类别名称和颜色
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        color = colors[class_id % len(colors)]
        
        # 创建矩形框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加类别标签
        ax.text(
            bbox[0], bbox[1] - 5,
            f'{class_name}',
            fontsize=10,
            color=color,
            fontproperties=chinese_font if chinese_font else None,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0),
            alpha=0.2
        )
    
    ax.set_title(f'Ground Truth Labels - {os.path.basename(image_path)}', 
                fontproperties=chinese_font if chinese_font else None)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"真实标签可视化已保存到: {save_path}")
    
    plt.show()

def visualize_predictions(image_path, model_results, class_names, save_path=None):
    """可视化预测结果"""
    # 读取图像
    if type(image_path) is str:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
    else:
        img = image_path
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # 创建matplotlib图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    # 处理预测结果
    if model_results and len(model_results) > 0:
        result = model_results[0]  # 取第一个结果
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            print(f"检测到 {len(boxes)} 个目标")
            
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                class_id = int(cls)
                class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                color = colors[class_id % len(colors)]
                
                # 创建矩形框
                rect = patches.Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1],
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加类别标签和置信度
                label_text = f'{class_name}: {conf:.2f}'
                ax.text(
                    box[0], box[1] - 5,
                    label_text,
                    fontsize=10,
                    color=color,
                    fontproperties=chinese_font if chinese_font else None,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0),
                    weight='bold',
                    alpha=0.2
                )
                
                # 打印检测信息
                print(f"目标 {i+1}: {class_name}, 置信度: {conf:.2f}, 位置: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        else:
            print("未检测到任何目标")
    else:
        print("预测结果为空")
    
    ax.set_title(f'Model Predictions - {os.path.basename(image_path)}', 
                fontproperties=chinese_font if chinese_font else None,
                fontsize=14,
                weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果可视化已保存到: {save_path}")
    
    plt.show()

def visualize_predictions_with_threshold(image_path, model_results, class_names, confidence_threshold=0.5, save_path=None):
    """可视化预测结果（带置信度阈值过滤）"""
    if type(image_path) is str:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
    else:
        img = image_path
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # 创建matplotlib图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    filtered_count = 0
    total_count = 0
    
    # 处理预测结果
    if model_results and len(model_results) > 0:
        result = model_results[0]  # 取第一个结果
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            total_count = len(boxes)
            
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                if conf >= confidence_threshold:  # 置信度过滤
                    filtered_count += 1
                    class_id = int(cls)
                    class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                    color = colors[class_id % len(colors)]
                    
                    # 创建矩形框
                    rect = patches.Rectangle(
                        (box[0], box[1]), 
                        box[2] - box[0], 
                        box[3] - box[1],
                        linewidth=2, 
                        edgecolor=color, 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加类别标签和置信度
                    label_text = f'{class_name}: {conf:.2f}'
                    ax.text(
                        box[0], box[1] - 5,
                        label_text,
                        fontsize=10,
                        color=color,
                        fontproperties=chinese_font if chinese_font else None,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0),
                        weight='bold',
                        alpha=0.2
                    )
            
            print(f"总检测目标: {total_count}, 置信度 >= {confidence_threshold} 的目标: {filtered_count}")
        else:
            print("未检测到任何目标")
    else:
        print("预测结果为空")
    
    title = f'Model Predictions (Conf >= {confidence_threshold}) - {os.path.basename(image_path)}'
    ax.set_title(title, 
                fontproperties=chinese_font if chinese_font else None,
                fontsize=14,
                weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果可视化已保存到: {save_path}")
    
    plt.show()


def visualize_predictions_and_gt(image_path, label_path, model_results, class_names, true_class_names, save_path=None):
    """同时可视化预测结果和真实标签"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # 读取真实标签
    gt_labels = read_yolo_labels(label_path, img_width, img_height)
    
    # 创建subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 颜色映射
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    # 左边显示真实标签
    ax1.imshow(img_rgb)
    for label in gt_labels:
        bbox = label['bbox']
        class_id = label['class_id']
        class_name = true_class_names[class_id] if class_id < len(true_class_names) else f'Class_{class_id}'
        color = colors[class_id % len(colors)]
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax1.add_patch(rect)
        
        ax1.text(
            bbox[0], bbox[1] - 5,
            f'GT: {class_name}',
            fontsize=8,
            color=color,
            fontproperties=chinese_font if chinese_font else None,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0),
            alpha=0.2
        )
    
    ax1.set_title('Ground Truth Labels', fontproperties=chinese_font if chinese_font else None)
    ax1.axis('off')
    
    # 右边显示预测结果
    ax2.imshow(img_rgb)
    
    # 处理预测结果
    if model_results and len(model_results) > 0:
        result = model_results[0]  # 取第一个结果
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                class_id = int(cls)
                class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                color = colors[class_id % len(colors)]
                
                rect = patches.Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1],
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none',
                    linestyle='--'  # 虚线表示预测
                )
                ax2.add_patch(rect)
                
                ax2.text(
                    box[0], box[1] - 5,
                    f'Pred: {class_name} ({conf:.2f})',
                    fontsize=8,
                    color=color,
                    fontproperties=chinese_font if chinese_font else None,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0),
                    alpha=0.2
                )
    
    ax2.set_title('Model Predictions', fontproperties=chinese_font if chinese_font else None)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比可视化已保存到: {save_path}")

def convert_to_negative(image, preserve_dtype=False):
    """
    将图像转换为负片，支持多种数据类型
    
    Args:
        image: 输入图像（路径或numpy数组）
        preserve_dtype: 是否保持原始数据类型
    """
    if isinstance(image, str):  # 如果输入是路径
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:  # 如果输入是numpy数组
        img = image.copy()
    
    if img is None:
        raise ValueError(f"无法读取图像: {image}")
    
    print(f"原始图像信息: 形状={img.shape}, 数据类型={img.dtype}, 最大值={img.max()}, 最小值={img.min()}")
    
    if preserve_dtype:
        # 保持原始数据类型
        if img.dtype == np.uint8:
            negative_img = 255 - img
        elif img.dtype == np.uint16:
            negative_img = 65535 - img
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                negative_img = 1.0 - img
            else:
                negative_img = img.max() - img
        else:
            raise ValueError(f"不支持的数据类型: {img.dtype}")
        
        print(f"保持原始类型的负片: {negative_img.dtype}")
        return negative_img
    else:
        # 统一转换为uint8（推荐，兼容YOLO）
        if img.dtype == np.uint8:
            negative_img = 255 - img
        elif img.dtype == np.uint16:
            # uint16 -> uint8 负片转换
            max_val = 65535
            negative_img_16 = max_val - img
            negative_img = (negative_img_16 / 256).astype(np.uint8)
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                negative_float = 1.0 - img
                negative_img = (negative_float * 255).astype(np.uint8)
            else:
                img_norm = img / img.max()
                negative_float = 1.0 - img_norm
                negative_img = (negative_float * 255).astype(np.uint8)
        else:
            # 其他类型统一处理
            img_norm = (img - img.min()) / (img.max() - img.min())
            negative_float = 1.0 - img_norm
            negative_img = (negative_float * 255).astype(np.uint8)
        
        print(f"转换为uint8负片: 形状={negative_img.shape}, 数据类型={negative_img.dtype}")
        return negative_img

def preprocess_image_for_yolo(image_path):
    """为YOLO预处理图像，支持多种数据类型"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print(f"原始图像: 形状={img.shape}, 数据类型={img.dtype}")
    
    # 数据类型转换
    if img.dtype == np.uint16:
        # uint16 -> uint8 转换
        if img.max() > 255:
            img = (img / 256).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        print("已转换为uint8格式")
    elif img.dtype not in [np.uint8]:
        # 其他类型转换为uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
        print("已转换为uint8格式")
    
    # 通道处理
    if len(img.shape) == 2:  # 灰度图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 1:  # 单通道图像
        img_rgb = np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # 已经是3通道
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"不支持的图像格式，通道数: {img.shape}")
    
    return img_rgb

def convert_dtype(image, target_dtype=np.uint8):
    """转换图像数据类型"""
    if isinstance(image, str):  # 如果输入是路径
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:  # 如果输入是numpy数组
        img = image.copy()
    
    if img is None:
        raise ValueError(f"无法读取图像: {image}")
    
    print(f"原始图像信息: 形状={img.shape}, 数据类型={img.dtype}, 最大值={img.max()}, 最小值={img.min()}")
    if img.dtype == target_dtype:
        return img

    if target_dtype == np.uint8:
        if img.dtype == np.uint16:
            if img.max() > 255:
                return (img / 256).astype(np.uint8)
            else:
                return img.astype(np.uint8)
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            else:
                return np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"不支持的转换到uint8的数据类型: {img.dtype}")

    elif target_dtype == np.uint16:
        if img.dtype == np.uint8:
            return (img.astype(np.uint16) * 256)
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                return (img * 65535).astype(np.uint16)
            else:
                img_norm = img / img.max()
                return (img_norm * 65535).astype(np.uint16)
        else:
            raise ValueError(f"不支持的转换到uint16的数据类型: {img.dtype}")
    
    elif target_dtype in [np.float32, np.float64]:
        if img.dtype == np.uint8:
            return (img.astype(target_dtype) / 255.0)
        elif img.dtype == np.uint16:
            return (img.astype(target_dtype) / 65535.0)
        else:
            raise ValueError(f"不支持的转换到浮点数的数据类型: {img.dtype}")

    else:
        raise ValueError(f"不支持的目标数据类型: {target_dtype}")


os.environ['CUDA_VISIBLE_DEVICES'] = '7'

USE_NEGATIVE = False  # 是否使用负片图像进行预测
USE_UINT16 = False
# BATCH_SIZE = 6
# IMG_SIZE = 2048
# PRETRAIN = True
# MODEL_SIZE = 'm'
# NUM_GPU = 2

model_name = "yolov8l_EMA_ECA_RTA_single_channel_2048_b4_2GPUs_300epochs"
# model_name = f"yolov8{MODEL_SIZE}_RTA_single_channel_{'pretrain_' if PRETRAIN else ''}{IMG_SIZE}_b{BATCH_SIZE}_{NUM_GPU}GPUs{'_uint16' if USE_UINT16 else ''}"

# 假设您的自定义配置文件在项目根目录的 'models' 文件夹下
custom_model_yaml = 'yolov8l_EMA_ECA.yaml' # <--- 替换成您真实的yaml文件路径

# 首先使用 .yaml 文件来构建正确的模型结构
# yolo = YOLO(custom_model_yaml, task="detect")

# # 然后将训练好的权重加载到这个结构正确的模型上
# yolo.load(f"./runs/detect/{model_name}/weights/best.pt")

yolo = YOLO(f"./runs/detect/{model_name}/weights/best.pt")

print("已成功加载自定义模型结构和权重。")


# 显示模型详细结构
print(yolo.model)


# 加载类别名称
train_class_names = load_class_names("datasets/yolo_RTA_tif/data.yaml")
print(f"类别名称: {train_class_names}")

pred_class_names = load_class_names("datasets/yolo_GuanWang_dataset_uint8/data.yaml")

# image_name = "DJ-RT-20220622-30.jpg"
# image_name = "YuanXing.png"
# 预处理图像
# image_path = "datasets/yolo_RTA/images/test/" + image_name

# processed_img = preprocess_image(image_path)

# image_path = "datasets/" + image_name
# image_path = "datasets/GuanWang_split/813-unqualified/813-UQ-01/813-UQ-01_part01.tif"
image_path = "datasets/yolo_GuanWang_dataset_uint8/images/train/1219-UQ-61_seg09.tif"
image_name = image_path.split('/')[-1]

# output_dir = "runs/compare/" + image_name.replace('.jpg', '/')
output_dir = 'runs/other_datasets/' + image_name.replace('.jpg', '/').replace('.png', '/').replace('.tif', '/')
os.makedirs(output_dir, exist_ok=True)

# 构建对应的标签文件路径
label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt').replace('.tif', '.txt')
if os.path.exists(label_path):
    print(f"标签文件路径: {label_path}")
else:
    print(f"未找到标签文件: {label_path}")


# 进行预测
if USE_NEGATIVE:
    negative_img = convert_to_negative(image_path, preserve_dtype=USE_UINT16)
    result = yolo(source=negative_img, save=False, conf=0.01)
    # image_path = negative_img  # 用于可视化
else:
    image = convert_dtype(image_path, target_dtype=np.uint8 if not USE_UINT16 else np.uint16)
    result = yolo(source=image, save=False, conf=0.01)

# 可视化预测结果
print("可视化预测结果...")
visualize_predictions(image_path, result, train_class_names, f"{output_dir}{model_name}_{'negative' if USE_NEGATIVE else 'original'}_predictions.png")

# 可视化预测结果（带置信度过滤）
print("可视化预测结果（置信度 >= 0.1）...")
visualize_predictions_with_threshold(image_path, result, train_class_names, 0.1, f"{output_dir}{model_name}_{'negative' if USE_NEGATIVE else 'original'}_predictions_filtered.png")


if os.path.exists(label_path):
    #可视化真实标签
    print("可视化真实标签...")
    visualize_ground_truth(image_path, label_path, pred_class_names, f"{output_dir}{model_name}_ground_truth_visualization.png")

    # 可视化预测结果和真实标签的对比
    print("可视化预测结果和真实标签对比...")
    visualize_predictions_and_gt(image_path, label_path, result, train_class_names, pred_class_names, f"{output_dir}{model_name}_prediction_vs_gt.png")


