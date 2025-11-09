from ultralytics import YOLO
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import OrderedDict

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号


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



def create_model(config_path, custom=True, pretrain=False):
    """
    创建YOLO模型。如果使用预训练权重，则通过修改权重字典的键来适应插入的新层。
    """
    if custom and pretrain:
        # 从配置路径中提取基础模型尺寸，如从'yolov8l_EMA_ECA.yaml'中提取'yolov8l'
        base_model_size = config_path.split('.')[0].split('_')[0]
        print(f"使用 {base_model_size}.pt 的预训练权重创建模型...")

        # 1. 加载标准模型的权重字典
        print("加载标准模型权重...")
        standard_model = YOLO(f'{base_model_size}.pt')
        standard_state_dict = standard_model.state_dict()

        # 2. 创建一个新的有序字典，用于存放修改后的键值对
        new_state_dict = OrderedDict()
        
        # 3. 定义新模块的插入位置。您说是在第9层之后，所以原第9层及之后的层索引都需要+1
        insertion_index = 9
        print(f"正在修改权重键名：将索引 >= {insertion_index} 的层号加1...")

        # 4. 遍历标准权重字典，修改键名
        for key, value in standard_state_dict.items():
            # 将键名按'.'分割，例如 'model.model.9.cv1.conv.weight' -> ['model', 'model', '9', 'cv1', 'conv', 'weight']
            parts = key.split('.')
            
            # 检查键名是否包含层索引（通常在第三个位置，并且是数字）
            if len(parts) > 2 and parts[2].isdigit():
                layer_idx = int(parts[2])
                
                # 如果层索引大于或等于插入点，则将其加1
                if layer_idx >= insertion_index:
                    new_idx = layer_idx + 1
                    parts[2] = str(new_idx)
                    new_key = '.'.join(parts)
                    new_state_dict[new_key] = value
                    # print(f"  映射: {key} -> {new_key}") # 取消注释以查看详细映射
                else:
                    # 索引小于插入点，键名保持不变
                    new_state_dict[key] = value
            else:
                # 不包含层索引的键（例如模型顶层的某些参数），直接复制
                new_state_dict[key] = value

        # 5. 根据你的自定义YAML文件创建模型结构
        print(f"根据 {config_path} 创建自定义模型结构...")
        model = YOLO(config_path)

        # 6. 加载修改后的权重字典
        print("加载修改后的权重...")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # # 7. (重要) 单独处理第一层，以适应单通道输入
        # print("正在修改第一层以适应单通道输入...")
        # # 获取标准模型第一层的权重
        # original_conv_weight = standard_model.model.model[0].conv.weight.data
        # # 计算均值以从3通道转换为1通道
        # single_channel_weight = original_conv_weight.mean(dim=1, keepdim=True)
        
        # # 将计算出的单通道权重赋给自定义模型的第一层
        # first_layer = model.model.model[0].conv
        # if first_layer.in_channels == 1:
        #     first_layer.weight.data = single_channel_weight
        #     print("  ✓ 成功修改第一层卷积权重为单通道。")
        # else:
        #     print(f"  ⚠️ 警告：自定义模型的第一层输入通道不是1（而是{first_layer.in_channels}），无法适配单通道权重。")

        # 检查权重加载情况
        print("检查权重加载情况...")
        print(f"Total keys {len(standard_state_dict)}")
        print(f"Loaded keys: {len(standard_state_dict) - len(missing_keys)}")
        print(f"Missing keys: {set(missing_keys)}")
        print(f"Unexpected keys: {set(unexpected_keys)}")

        print("\n模型创建和预训练权重加载完成！")
    
    elif not custom and pretrain:
        print("直接使用标准模型的预训练权重创建模型...")
        model_size = config_path.split('.')[0].split('_')[0]  # 假设config_path是类似
        model = YOLO(f'{model_size}.pt')

    else:
        print("使用随机初始化权重创建模型...")
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


gpu_list = '4,5,6'
# 只使用GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
gpu_num = len(gpu_list.split(','))

dataset = 'yolo_RTA_tif'
# dataset = 'yolo_GuanWang_dataset_uint8'
data_yaml = f'datasets/{dataset}/data_origin.yaml'
imgsz = 1800  # 输入图像尺寸
batch_size = 6  # 批量大小
uint16flag = False  # 是否使用uint16数据
dtype = '_uint16' if uint16flag else ''
pretrain = True  # 是否使用预训练权重
custom = True  # 是否使用自定义模型结构
epochs = 300  # 训练轮数

# 检查GPU可用性
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 创建单通道输入的YOLO模型
# print("正在创建单通道YOLO模型...")
# model_name = 'yolov8l_EMA_ECA'  # 模型大小，可以是'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
model_name = 'yolo11l_EMA_ECA'  # 使用自定义的YOLOv11模型结构
model = create_model(f'{model_name}.yaml', custom=custom, pretrain=pretrain)

# 验证模型结构
verify_model_structure(model)



# 开始训练
print("开始训练...")
model.train(
    data=data_yaml, 
    epochs=epochs, 
    imgsz=imgsz, 
    batch=batch_size,
    project='runs_' + dataset, 
    name=f'{model_name}_RTA_single_channel_{"pretrain_" if pretrain else ""}{"custom_" if custom else ""}{imgsz}_b{batch_size}_{torch.cuda.device_count()}GPUs{dtype}_{epochs}epochs', 
    amp=False,  # 禁用AMP
    verbose=True  # 显示详细训练信息
)