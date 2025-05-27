import os
import cv2
import argparse
import shutil
from os.path import basename
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import glob
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"

def make_parser():
    parser = argparse.ArgumentParser("EfficientNet process!")
    parser.add_argument("--test_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--ori_img_dir",type=str,default=None)
    parser.add_argument("--threshold_B",type=float,default=0.97)
    parser.add_argument("--threshold_S",type=float,default=0.9)
    parser.add_argument("--weightB",type=str,default="tools/Step6_classify/weights/weightB/0804_1338/Epoch32.pth",help="large-cell model path")
    parser.add_argument("--weightS",type=str,default="tools/Step6_classify/weights/weightS/0803_1049/Epoch30.pth",help="small-cell model path")
    parser.add_argument("--imgszB",type=int,default=224,help="test image size")
    parser.add_argument("--imgszS",type=int,default=90,help="test image size")
    # opt=parser.parse_known_args()[0]
    return parser

class Test_model():
    def __init__(self,opt):
        self.imgszB=opt.imgszB
        self.imgszS=opt.imgszS
        self.img_dir=opt.test_dir
        self.save_dir=opt.save_dir
        self.ori_img_dir=opt.ori_img_dir
        self.threshold_B=opt.threshold_B
        self.threshold_S=opt.threshold_S
        self.weightB=opt.weightB
        self.modelB=(torch.load(opt.weightB)).to(device)
        self.modelB.eval()
        self.weightS=opt.weightS
        self.modelS=(torch.load(opt.weightS)).to(device)
        self.modelS.eval()
        self.class_name=['cancer','normal']

  
    def __call__(self):
        data_transorformB=torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.imgszB,self.imgszB)),
                torchvision.transforms.CenterCrop((self.imgszB,self.imgszB)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        data_transorformS=torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.imgszS,self.imgszS)),
                torchvision.transforms.CenterCrop((self.imgszS,self.imgszS)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        img_list=glob.glob(self.img_dir+os.sep+"*.png")
        print(img_list[0])
        print(len(img_list))
        big_cancer=0
        small_cancer=0
        normal=0
        big = 0
        small = 0
        destination_file = new_path = os.path.join(os.path.dirname(self.save_dir), os.path.basename(self.save_dir) + '.txt')
        with open( destination_file, 'w') as file:
            file.write(f"image path：{self.img_dir}\n")
            file.write(f"ori image path：{self.ori_img_dir}\n")
            file.write(f"weightB：{self.weightB}\n")
            file.write(f"threshold_B:{self.threshold_B}\n")
            file.write(f"weightS：{self.weightS}\n")
            file.write(f"threshold_S:{self.threshold_S}\n")
            file.write("Image:\tpred:\tprob：\timg_type:\n")

            img_destination_directory = os.path.join(self.save_dir,"cancer")
            os.makedirs(img_destination_directory,exist_ok=True)
            img_destination_directory_ori = os.path.join(self.save_dir,"cancer_ori")
            os.makedirs(img_destination_directory_ori,exist_ok=True)

            for imgpath in tqdm(img_list, desc="Processing Images"):
                img0=cv2.imread(imgpath)
                h,w=img0.shape[:2]
                if h<30 or w<30:continue

                img=self.expend_img(img0)
                img=Image.fromarray(img)
                
                
                if h > 90 or w > 90:
                    img=data_transorformB(img)
                    img=torch.reshape(img, (-1, 3, self.imgszB, self.imgszB)).to(device)
                    pred=self.modelB(img)
                    pos_prob_threshold = self.threshold_B
                    img_type = "Big"
                    big+=1

                else:
                    img=data_transorformS(img)
                    img=torch.reshape(img, (-1, 3, self.imgszS, self.imgszS)).to(device)
                    pred=self.modelS(img)
                    pos_prob_threshold = self.threshold_S
                    img_type = "Small"
                    small+=1

                pos_prob = pred[0][0].item()

                if pos_prob > pos_prob_threshold:
                    outputs = self.class_name[0]
                else:
                    outputs = self.class_name[1]
                    
                if outputs=="cancer":
                    if img_type == "Big":
                        big_cancer+=1
                    else : small_cancer+=1
                    
                    img_save_name = f"{(pred[0][0].item()):.4f}_"+basename(imgpath)
                    img_save_path = os.path.join(img_destination_directory, img_save_name)
                    img_save_path_ori = os.path.join(img_destination_directory_ori, img_save_name)
                    imgpath_ori = os.path.join(self.ori_img_dir,basename(imgpath))
                    shutil.copy(imgpath, img_save_path)
                    shutil.copy(imgpath_ori, img_save_path_ori)
                    file.write(f"{basename(imgpath)}\t{outputs}\t{str(pred[0][0].item())}\t{img_type}\n")
            
                else: 
                    normal+=1
                    file.write(f"{basename(imgpath)}\t{outputs}\t{str(pred[0][0].item())}\t{img_type}\n")
                    
                
            print(f"big_cancer={big_cancer}\tsmall_cancer={small_cancer}\tbig={big}\tsmall={small}\ttotal={big+small}\n")
            file.write(f"big_cancer={big_cancer}\tsmall_cancer={small_cancer}\tbig={big}\tsmall={small}\ttotal={big+small}\n")

    
    def expend_img(self,img,fill_pix=255):
        def adjust_image(image, gamma=1.1, brightness=0.93, contrast=1.01, r_gain=1.04, g_gain=1.02, b_gain=1.02):
            """
            调整图像的伽玛值、亮度、对比度和RGB三通道独立增益
            - gamma: 伽玛值 (0.6)
            - brightness: 亮度 (100%)
            - contrast: 对比度 (92%)
            - r_gain: 红色通道增益 (100%)
            - g_gain: 绿色通道增益 (100%)
            - b_gain: 蓝色通道增益 (100%)
            """
            # 伽玛值调整
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_gamma = cv2.LUT(image, table)
            
            # 亮度调整
            img_bright = cv2.convertScaleAbs(img_gamma, alpha=brightness, beta=0)
            
            # 对比度调整
            img_contrast = cv2.convertScaleAbs(img_bright, alpha=contrast, beta=0)
            
            # 三通道独立增益调整
            # 分离通道
            b, g, r = cv2.split(img_contrast)
            
            # 应用增益
            r = cv2.convertScaleAbs(r, alpha=r_gain, beta=0)
            g = cv2.convertScaleAbs(g, alpha=g_gain, beta=0)
            b = cv2.convertScaleAbs(b, alpha=b_gain, beta=0)
            
            # 合并通道
            img_rgb = cv2.merge([b, g, r])
            
            return img_rgb
      
        fill_pix=[255,255,255] 
        h,w=img.shape[:2]

        if h>=w: 
            padd_width=int(h-w)//2
            padd_top,padd_bottom,padd_left,padd_right=0,0,padd_width,padd_width 
        elif h<w: 
            padd_high=int(w-h)//2
            padd_top,padd_bottom,padd_left,padd_right=padd_high,padd_high,0,0 
        
        new_img = adjust_image(img, gamma=1.1, brightness=0.93, contrast=1.01, r_gain=1.04, g_gain=1.02, b_gain=1.02)
        new_img = cv2.copyMakeBorder(new_img,padd_top,padd_bottom,padd_left,padd_right,cv2.BORDER_CONSTANT, value=fill_pix)
        image_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    


if __name__ == '__main__':
    args = make_parser().parse_args()
    run_dir = args.test_dir
    file_name = os.path.basename(run_dir)
    
    save_dir =args.save_dir
    args.save_dir = save_dir
    weightB = args.weightB
    threshold_B = args.threshold_B
    weightS = args.weightS
    threshold_S = args.threshold_S
    print("!!!!!!!!!!!!! Details !!!!!!!!!!!!!")
    print("save_dir:" + save_dir)
    print("weightB:" + weightB )
    print("threshold_B:" + str(threshold_B))
    print("weightS:" + weightS )
    print("threshold_S:" + str(threshold_S))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    test_img=Test_model(args)
    test_img()
