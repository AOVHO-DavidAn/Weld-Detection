import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm


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


def main():

    image_root = 'datasets/yolo_RTA/images'
    output_root = 'datasets/yolo_RTA/visualizations'

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历所有图像文件
    for image_file in os.listdir(image_root):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_root, image_file)
            label_path = os.path.join('datasets/yolo_RTA/labels', image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            visualize_ground_truth(image_path, label_path, class_names, save_path=os.path.join(output_root, image_file))