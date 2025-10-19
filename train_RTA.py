from ultralytics import YOLO
import torch
import os

def create_single_channel_model(config_path, pretrain=False):
    """创建支持单通道输入的YOLO模型"""
    # 从配置文件创建模型
    if pretrain:
        model = YOLO('datasets/yolov8n.pt') # 使用预训练权重
    else:
        model = YOLO(config_path)
    
    
    return model

def verify_model_structure(model):
    """验证模型结构"""
    print("\n=== 验证模型结构 ===")
    conv_count = 0
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and conv_count < 3:
            print(f"卷积层 {conv_count+1} ({name}):")
            print(f"  输入通道: {module.in_channels}")
            print(f"  输出通道: {module.out_channels}")
            print(f"  卷积核大小: {module.kernel_size}")
            print(f"  步长: {module.stride}")
            
            # 检查权重分布
            weight_mean = module.weight.data.mean().item()
            weight_std = module.weight.data.std().item()
            print(f"  权重统计: 均值={weight_mean:.6f}, 标准差={weight_std:.6f}")
            
            if 0.001 < weight_std < 1.0:
                print(f"  ✓ 权重分布正常（随机初始化）")
            else:
                print(f"  ⚠️ 权重分布异常")
            
            conv_count += 1
    print("=== 验证完成 ===\n")


gpu_list = '2'
# 只使用GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
# gpu_num = len(gpu_list.split(','))

imgsz = 4096  # 输入图像尺寸
batch_size = 1  # 批量大小
uint16flag = False  # 是否使用uint16数据
dtype = '_uint16' if uint16flag else ''
pretrain = False  # 是否使用预训练权重

# 检查GPU可用性
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 创建单通道输入的YOLO模型
print("正在创建单通道YOLO模型...")
model_size = 'yolov8n'
model = create_single_channel_model(f'{model_size}.yaml')

# 验证模型结构
verify_model_structure(model)



# 开始训练
print("开始训练...")
model.train(
    data=f'datasets/yolo_RTA_tif{dtype}/data.yaml', 
    epochs=300, 
    imgsz=imgsz, 
    batch=batch_size, 
    name=f'{model_size}_RTA_single_channel_{"pretrain_" if pretrain else ""}{imgsz}_b{batch_size}_{torch.cuda.device_count()}GPUs{dtype}', 
    amp=False,  # 禁用AMP
    verbose=True  # 显示详细训练信息
)