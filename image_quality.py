import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无GUI后端
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import sys

class IQIAnalyzer:
    def __init__(self, image_path, output_dir):
        # 读取TIFF图像（支持8位/16位）
        self.image = cv2.imread(image_path, -1)  # 读取原始深度
        if self.image is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}（路径或格式错误）")
        
        # 转换为灰度图（若为多通道）
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 16位TIFF归一化到8位
        if self.image.dtype == np.uint16:
            self.image = cv2.normalize(
                self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        
        self.original_image = self.image.copy()
        self.ROI = None
        self.line_pairs = []
        self.dip_results = {}
        self.output_dir = output_dir  # 结果保存目录
        os.makedirs(self.output_dir, exist_ok=True)  # 确保目录存在

    def locate_iqi(self, min_area=5000, max_area=100000):
        """自动定位像质计区域（服务器无GUI，仅支持自动定位）"""
        blurred = cv2.GaussianBlur(self.image, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                self.ROI = self.image[y:y+h, x:x+w]
                cv2.rectangle(self.original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print(f"像质计定位成功：区域坐标 ({x},{y})-({x+w},{y+h})")
                # 保存定位结果图像（修正缩进：与上一行保持一致的4个空格）
                定位结果路径 = os.path.join(self.output_dir, "iqi_location.png")
                cv2.imwrite(定位结果路径, self.original_image)
                print(f"定位结果已保存至：{定位结果路径}")
                return True
        
        # 服务器无GUI，自动定位失败则终止程序
        raise RuntimeError("自动定位像质计失败，请检查图像质量或调整定位参数（min_area/max_area）")

    def detect_line_pairs(self, min_peak_distance=10, peak_prominence=50):
        if self.ROI is None:
            raise ValueError("请先定位像质计区域")
        
        rows, cols = self.ROI.shape
        step = max(1, rows // 21)  # 确保取21行
        selected_rows = self.ROI[::step][:21]
        avg_profile = np.mean(selected_rows, axis=0)
        
        # 检测峰值（可根据实际图像调整参数）
        peaks, _ = find_peaks(avg_profile, distance=min_peak_distance, prominence=peak_prominence)
        
        # 组成线对
        for i in range(0, len(peaks)-1, 2):
            if i+1 >= len(peaks):
                break
            self.line_pairs.append((peaks[i], peaks[i+1]))
        
        print(f"检测到 {len(self.line_pairs)} 组线对")
        return self.line_pairs

    def calculate_dip(self):
        if not self.line_pairs:
            raise ValueError("未检测到线对，请调整线对检测参数")
        
        rows, cols = self.ROI.shape
        step = max(1, rows // 21)
        selected_rows = self.ROI[::step][:21]
        avg_profile = np.mean(selected_rows, axis=0)
        
        for idx, (peak1, peak2) in enumerate(self.line_pairs, 1):
            A = avg_profile[peak1]
            B = avg_profile[peak2]
            valley_start = min(peak1, peak2) + 1
            valley_end = max(peak1, peak2) - 1
            C = np.min(avg_profile[valley_start:valley_end]) if valley_start < valley_end else (A + B)/2
            
            dip = 100 * (A + B - 2 * C) / (A + B) if (A + B) != 0 else 0
            self.dip_results[f"D{idx}"] = round(dip, 2)
        
        return self.dip_results

    def get_resolution_result(self, threshold=20):
        if not self.dip_results:
            raise ValueError("请先计算下沉值")
        
        sorted_pairs = sorted(self.dip_results.items(), key=lambda x: int(x[0][1:]))
        for pair_id, dip in sorted_pairs:
            if dip < threshold:
                return f"可分辨极限线对：{pair_id}（下沉值：{dip}%）"
        return f"所有线对下沉值均≥20%，极限线对为最后一组：{sorted_pairs[-1][0]}"

    def visualize(self):
        """生成可视化结果并保存为图片（不显示，直接保存）"""
        plt.figure(figsize=(12, 8))
        
        # 1. 像质计定位结果
        plt.subplot(211)
        plt.imshow(self.original_image, cmap="gray")
        plt.title("像质计定位结果")
        
        # 2. 灰度剖面与线对峰值
        if self.line_pairs:
            rows, cols = self.ROI.shape
            step = max(1, rows // 21)
            selected_rows = self.ROI[::step][:21]
            avg_profile = np.mean(selected_rows, axis=0)
            
            plt.subplot(212)
            plt.plot(avg_profile, label="21行平均灰度剖面")
            for idx, (peak1, peak2) in enumerate(self.line_pairs, 1):
                plt.scatter([peak1, peak2], [avg_profile[peak1], avg_profile[peak2]], 
                           c="red", label=f"D{idx}线对")
            plt.axhline(y=0, color='r', linestyle='--', label="20%阈值参考线")
            plt.title("线对灰度剖面与峰值检测")
            plt.legend()
        
        plt.tight_layout()
        # 保存可视化结果
        可视化路径 = os.path.join(self.output_dir, "iqi_analysis.png")
        plt.savefig(可视化路径, dpi=300)
        plt.close()  # 关闭画布释放内存
        print(f"可视化结果已保存至：{可视化路径}")

    def save_results(self):
        """将分析结果保存为文本文件"""
        结果路径 = os.path.join(self.output_dir, "iqi_result.txt")
        with open(结果路径, "w", encoding="utf-8") as f:
            f.write("双线型像质计分析结果\n")
            f.write("="*30 + "\n")
            f.write("下沉值计算结果：\n")
            for pair_id, dip in self.dip_results.items():
                f.write(f"{pair_id}: {dip}%\n")
            f.write("\n" + self.get_resolution_result() + "\n")
        print(f"分析结果已保存至：{结果路径}")


if __name__ == "__main__":
    # 固定图像路径（用户指定的TIFF路径）
    image_path = "/home/yukun/mycode/RTAnormaly/Weld-Detection/datasets/GuanWang_split/images_negative_vis/813-I-074_seg02.tif"
    image_name = image_path.split('/')[-1].split('.')[0]
    # 结果输出目录（保存在图像同目录下的results文件夹）
    output_dir = "./runs/image_quality_results" + f"/{image_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化分析器
        analyzer = IQIAnalyzer(image_path, output_dir)
        # 自动定位像质计
        analyzer.locate_iqi()
        # 检测线对（可根据图像调整参数）
        analyzer.detect_line_pairs(min_peak_distance=5, peak_prominence=30)
        # 计算下沉值
        analyzer.calculate_dip()
        # 保存结果文本
        analyzer.save_results()
        # 生成并保存可视化图像
        analyzer.visualize()
        
        print("\n分析完成！所有结果已保存至：", output_dir)
    except Exception as e:
        print(f"运行出错：{str(e)}", file=sys.stderr)
        sys.exit(1)