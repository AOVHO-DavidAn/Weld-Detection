#!/usr/bin/env python3
"""
简单的中文字体解决方案
下载并使用思源黑体
"""

import os
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def download_chinese_font():
    """下载中文字体文件"""
    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    font_path = "SourceHanSansSC-Regular.otf"
    
    if not os.path.exists(font_path):
        print("正在下载中文字体...")
        try:
            urllib.request.urlretrieve(font_url, font_path)
            print(f"字体下载完成: {font_path}")
        except Exception as e:
            print(f"字体下载失败: {e}")
            return None
    else:
        print(f"字体文件已存在: {font_path}")
    
    return font_path

def setup_matplotlib_chinese():
    """设置matplotlib中文支持"""
    # 方法1: 下载字体文件
    font_path = download_chinese_font()
    
    if font_path and os.path.exists(font_path):
        # 使用下载的字体
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"使用下载的字体: {font_path}")
        return prop
    
    # 方法2: 使用系统字体
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVu-Sans.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            print(f"使用系统字体: {font_path}")
            return prop
    
    # 方法3: 设置默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用默认字体设置")
    return None

def test_chinese_display():
    """测试中文显示"""
    chinese_font = setup_matplotlib_chinese()
    
    plt.figure(figsize=(10, 6))
    
    # 测试文本
    test_texts = ['气孔', '未熔透', '未熔合', '裂纹', '夹渣', '伪缺陷', '焊缝']
    
    for i, text in enumerate(test_texts):
        plt.text(0.1 + (i % 4) * 0.2, 0.7 - (i // 4) * 0.3, text, 
                fontsize=16, fontproperties=chinese_font if chinese_font else None)
    
    plt.title('中文字体测试', fontproperties=chinese_font if chinese_font else None)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.savefig('chinese_font_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("中文字体测试完成，图片保存为 chinese_font_test.png")

if __name__ == "__main__":
    test_chinese_display()
