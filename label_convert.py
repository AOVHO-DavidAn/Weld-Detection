import json
import tqdm
import os
import numpy as np

def convert_labelme_to_yolo_bbox(json_file, output_dir, class_mapping):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    yolo_annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_mapping:
            # Add new class with next available ID
            if label == None or label.strip() == "":
                label = "无缺陷"
            next_id = max(class_mapping.values()) + 1 if class_mapping else 0
            class_mapping[label] = next_id
            print(f"Added new class: '{label}' with ID {next_id}")
            
        class_id = class_mapping[label]
        points = np.array(shape['points'])
        
        # 计算边界框
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # 转换为YOLO格式（归一化坐标）
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 保存YOLO格式标注
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return class_mapping


def batch_convert_labelme_to_yolo():
    input_dir = "data/Raw_data/json"
    output_dir = "data/Raw_data_jpg/labels"

    os.makedirs(output_dir, exist_ok=True)
    
    # class_mapping = {
    #     '气孔': 0,
    #     '未熔透': 1,
    #     '未熔合': 2,
    #     '裂纹': 3,
    #     '夹渣': 4,
    #     '伪缺陷': 5,
    #     '焊缝': 6
    # }
    class_mapping = {}

    for json_file in tqdm.tqdm(os.listdir(input_dir), desc="Converting JSON to YOLO"):
        if json_file.endswith('.json'):
            json_path = os.path.join(input_dir, json_file)
            class_mapping = convert_labelme_to_yolo_bbox(json_path, output_dir, class_mapping)
            # print(f"Converted: {json_file}")
    
    return class_mapping

# 生成classes.txt文件
def create_classes_file(class_mapping, output_path):
    classes = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    print("Classes to be written to classes.txt:", classes)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(classes))
        

if __name__ == "__main__":
    class_mapping = batch_convert_labelme_to_yolo()
    # After conversion, save the classes.txt file
    
    print("Final class mapping:", class_mapping)
    
    create_classes_file(class_mapping, "data/Raw_data_jpg/classes.txt")