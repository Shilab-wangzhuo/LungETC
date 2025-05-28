import os
import cv2
import argparse
import shutil
from os.path import basename
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import glob
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_parser():
    parser = argparse.ArgumentParser("DenseNet161 Classifier")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ori_img_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight", type=str, default="tools/Step7_SCLC_NSCLC_classification/weight/densenet161.pth", help="DenseNet161 model path")
    parser.add_argument("--imgsz", type=int, default=224, help="test image size")
    return parser


# 定义DenseNet161模型
class CustomDenseNet161(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDenseNet161, self).__init__()
        # 加载DenseNet161模型
        self.densenet = models.densenet161(pretrained=False)
        
        # 修改最后的分类器以适应二分类任务
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)

class Test_model():
    def __init__(self, opt):
        self.imgsz = opt.imgsz
        self.img_dir = opt.test_dir
        self.save_dir = opt.save_dir
        self.ori_img_dir = opt.ori_img_dir
        self.threshold = opt.threshold
        self.weight = opt.weight
        
        # 加载模型
        print(f"Loading model from {self.weight}...")
        self.model = self.load_model(self.weight).to(device)
        self.model.eval()
        self.class_name = ['NSCLC', 'SCLC']  # 0=NSCLC, 1=SCLC
    
    def load_model(self, weight_path):
        # 创建模型实例
        model = CustomDenseNet161(num_classes=2)
        
        # 加载权重
        state_dict = torch.load(weight_path, map_location=device)
        
        # 检查是否需要处理state_dict (有时预训练模型的state_dict格式可能不同)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        
        model.load_state_dict(state_dict)
        return model
    
    def pad_to_square(self, image, fill_color=(255, 255, 255)):
        """将图像填充为正方形"""
        try:
            # 获取原始图像的大小
            width, height = image.size
            
            # 计算填充后的目标大小
            max_side = max(width, height)
            
            # 创建一个新的白色背景图像
            new_image = Image.new('RGB', (max_side, max_side), fill_color)
            
            # 将原始图像粘贴到新图像的中心
            new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
            
            return new_image
        except Exception as e:
            print(f"Error in pad_to_square: {e}")
            # 如果出错，返回一个空白的正方形图像
            return Image.new('RGB', (max(1, max_side), max(1, max_side)), fill_color)
    
    def __call__(self):
        # 图像预处理
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.imgsz, self.imgsz)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.8202, 0.8579, 0.9037], 
                                            std=[0.2687, 0.2223, 0.1526])
        ])
        
        img_list = glob.glob(self.img_dir + os.sep + "*.png")
        print(f"Found {len(img_list)} images in {self.img_dir}")
        
        sclc_count = 0
        nsclc_count = 0
        
        # 创建输出目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建SCLC子目录
        sclc_dir = os.path.join(self.save_dir, "SCLC")
        os.makedirs(sclc_dir, exist_ok=True)
        
        # 创建SCLC_ori子目录
        sclc_ori_dir = os.path.join(self.save_dir, "SCLC_ori")
        os.makedirs(sclc_ori_dir, exist_ok=True)
        
        # 创建NSCLC子目录
        nsclc_dir = os.path.join(self.save_dir, "NSCLC")
        os.makedirs(nsclc_dir, exist_ok=True)
        
        # 创建NSCLC_ori子目录
        nsclc_ori_dir = os.path.join(self.save_dir, "NSCLC_ori")
        os.makedirs(nsclc_ori_dir, exist_ok=True)
        
        # 创建结果文本文件
        result_file = os.path.join(os.path.dirname(self.save_dir), os.path.basename(self.save_dir) + '.txt')
        
        with open(result_file, 'w') as file:
            file.write(f"image path：{self.img_dir}\n")
            file.write(f"ori image path：{self.ori_img_dir}\n")
            file.write(f"weight：{self.weight}\n")
            file.write(f"threshold:{self.threshold}\n")
            file.write("Image:\tpred:\tprob：\n")
            
            for imgpath in tqdm(img_list, desc="Processing Images"):
                # 读取图像
                try:
                    img0 = cv2.imread(imgpath)
                    if img0 is None:
                        print(f"Warning: Could not read image {imgpath}")
                        continue
                    
                    h, w = img0.shape[:2]
                    if h < 30 or w < 30:
                        continue
                    
                    # 转换为PIL图像
                    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # 填充为正方形
                    pil_img = self.pad_to_square(pil_img)
                    
                    # 应用预处理
                    img_tensor = data_transform(pil_img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加batch维度
                    
                    # 模型预测
                    with torch.no_grad():
                        pred = self.model(img_tensor)
                        probs = torch.softmax(pred, dim=1)
                        nsclc_prob = probs[0][0].item()  # 索引0对应NSCLC
                        sclc_prob = probs[0][1].item()   # 索引1对应SCLC
                    
                    # 根据阈值确定类别
                    if sclc_prob > self.threshold:
                        output_class = self.class_name[1]  # SCLC
                        sclc_count += 1
                        
                        # 保存被分类为SCLC的图像
                        img_save_name = f"{sclc_prob:.4f}_" + basename(imgpath)
                        img_save_path = os.path.join(sclc_dir, img_save_name)
                        img_save_path_ori = os.path.join(sclc_ori_dir, img_save_name)
                        
                        # 复制图像到目标目录
                        shutil.copy(imgpath, img_save_path)
                        
                        # 复制原始图像
                        imgpath_ori = os.path.join(self.ori_img_dir, basename(imgpath))
                        if os.path.exists(imgpath_ori):
                            shutil.copy(imgpath_ori, img_save_path_ori)
                    else:
                        output_class = self.class_name[0]  # NSCLC
                        nsclc_count += 1
                        
                        # 保存被分类为NSCLC的图像
                        img_save_name = f"{nsclc_prob:.4f}_" + basename(imgpath)
                        img_save_path = os.path.join(nsclc_dir, img_save_name)
                        img_save_path_ori = os.path.join(nsclc_ori_dir, img_save_name)
                        
                        # 复制图像到目标目录
                        shutil.copy(imgpath, img_save_path)
                        
                        # 复制原始图像
                        imgpath_ori = os.path.join(self.ori_img_dir, basename(imgpath))
                        if os.path.exists(imgpath_ori):
                            shutil.copy(imgpath_ori, img_save_path_ori)
                    
                    # 写入结果文件
                    file.write(f"{basename(imgpath)}\t{output_class}\t{sclc_prob if output_class == 'SCLC' else nsclc_prob}\n")
                
                except Exception as e:
                    print(f"Error processing {imgpath}: {e}")
            
            # 写入统计信息
            total = sclc_count + nsclc_count
            file.write(f"\nSummary:\n")
            file.write(f"SCLC: {sclc_count}\n")
            file.write(f"NSCLC: {nsclc_count}\n")
            file.write(f"Total: {total}\n")
            
            print(f"\nSummary:")
            print(f"SCLC: {sclc_count}")
            print(f"NSCLC: {nsclc_count}")
            print(f"Total: {total}")

if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # 打印参数信息
    print("!!!!!!!!!!!!! Details !!!!!!!!!!!!!")
    print("test_dir:" + args.test_dir)
    print("save_dir:" + args.save_dir)
    print("weight:" + args.weight)
    print("threshold:" + str(args.threshold))
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 实例化测试模型并执行
    test_img = Test_model(args)
    test_img()