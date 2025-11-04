import os
from ultralytics import YOLO
import torch

# --- 1. 配置 (请根据您的实际情况修改) ---

# 您在 yolo_RTA_tif 数据集上训练好的模型权重路径
# 例如: 'runs/detect/yolov8n_RTA_.../weights/best.pt'
PRETRAINED_WEIGHTS = 'runs/detect/yolov8l_EMA_ECA_RTA_single_channel_2048_b4_2GPUs_300epochs/weights/best.pt' 

# 新的数据集 (yolo_GuanWang_dataset) 的 data.yaml 文件路径
NEW_DATASET_YAML = 'datasets/yolo_GuanWang_dataset_uint8/data.yaml'

# 微调训练的参数
EPOCHS = 200          # 微调通常不需要太多轮次
BATCH_SIZE = 2       # 根据您的 GPU 显存调整
IMG_SIZE = 2048      # 建议使用与新数据集匹配的图像尺寸
LEARNING_RATE = 1e-4 # 微调时建议使用较小的学习率
PROJECT_NAME = 'fine_tune_runs' # 保存微调实验的根目录
FREEZE = False
EXPERIMENT_NAME = f'GuanWang_finetune_epochs{EPOCHS}_freeze={FREEZE}' # 本次微调实验的名称
GPU_LIST = '2'

# --- 2. 主逻辑 (通常无需修改) ---

def fine_tune_yolo(freeze=False):
    """
    执行 YOLOv8 模型的微调，仅训练检测头。
    """
    # 检查预训练权重文件是否存在
    if not os.path.exists(PRETRAINED_WEIGHTS):
        print(f"错误: 找不到预训练权重文件: {PRETRAINED_WEIGHTS}")
        print("请确保路径正确，并指向一个 .pt 文件。")
        return

    # 检查新数据集的 yaml 文件是否存在
    if not os.path.exists(NEW_DATASET_YAML):
        print(f"错误: 找不到新数据集的 YAML 文件: {NEW_DATASET_YAML}")
        return

    # 步骤 1: 加载预训练模型
    print(f"成功加载预训练模型: {PRETRAINED_WEIGHTS}")
    model = YOLO(PRETRAINED_WEIGHTS)

    if freeze:
        # 步骤 2: 冻结模型的骨干和颈部层
        # YOLOv8 的标准模型结构中，0-21 层是骨干和颈部，第 22 层是检测头。
        # 我们将冻结前 22 层。
        print("正在冻结模型的骨干网络和颈部网络层...")
        
        # 获取模型的所有模块
        modules = model.model.model
        
        # 确定要冻结的层数。通常检测头是最后一个模块。
        # 为了更稳妥，我们冻结到倒数第二层。
        num_layers_to_freeze = len(list(modules)) - 1
        
        frozen_count = 0
        for i, module in enumerate(modules):
            if i < num_layers_to_freeze:
                # 遍历模块内的所有参数并设置 requires_grad = False
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += 1
        
        print(f"共冻结了 {frozen_count} 个模块。只有检测头将被训练。")

        # (可选) 验证冻结是否成功
        print("\n--- 权重梯度状态检查 (前5层和最后3层) ---")
        all_params = list(model.model.parameters())
        for i, p in enumerate(all_params):
            if i < 5 or i >= len(all_params) - 3:
                print(f"参数 {i}: requires_grad = {p.requires_grad}")
        print("--------------------------------------------\n")

        # 步骤 3: 开始微调训练
        # 使用 model.train() 方法，并传入新的数据集配置
        print("开始在新的数据集上微调检测头...")

    else:
        print("开始在整个模型上进行微调")

    model.train(
        data=NEW_DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE,  # 设置初始学习率
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        # 其他训练参数可以按需添加
        # 例如: device='0', workers=8, ...
        amp=False # 建议在微调时关闭 amp 以获得更稳定的梯度
    )

    print("\n微调完成！模型已保存在目录中: ")
    print(f"{PROJECT_NAME}/{EXPERIMENT_NAME}")

if __name__ == '__main__':
    # 检查 CUDA 可用性
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # gpu_list = '0'
        # 只使用GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_LIST
        print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    fine_tune_yolo(freeze=FREEZE)