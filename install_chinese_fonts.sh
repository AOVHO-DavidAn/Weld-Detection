#!/bin/bash

# 安装中文字体的脚本

echo "正在安装中文字体..."

# 更新包管理器
sudo apt update

# 安装中文字体包
sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei fonts-dejavu-core fonts-liberation

# 安装文泉驿微米黑字体
sudo apt install -y ttf-wqy-microhei

# 清除matplotlib缓存
python3 -c "import matplotlib.font_manager as fm; fm._rebuild()"

# 列出可用的中文字体
echo "可用的中文字体:"
fc-list :lang=zh-cn

echo "字体安装完成！"

# 验证字体是否可用
python3 -c "
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找中文字体
fonts = [f.name for f in fm.fontManager.ttflist if 'zh' in f.name.lower() or 'hei' in f.name.lower() or 'microhei' in f.name.lower()]
print('matplotlib中可用的中文字体:', fonts)

# 测试中文显示
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, '测试中文字体显示', fontsize=20, ha='center')
plt.title('中文字体测试')
plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
print('字体测试图片已保存为 font_test.png')
"
