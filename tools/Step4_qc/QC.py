import torch
import os
import torchvision
import glob
from PIL import Image
import cv2
import argparse
import shutil
from os.path import basename
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"

def make_parser():
    parser = argparse.ArgumentParser("EfficientNet process!")
    parser.add_argument("--test_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None) 
    parser.add_argument("--threshold",type=float,default=0.6)
    parser.add_argument("--weights",type=str,default="tools/Step4_qc/weights/0703_1409/Epoch21.pth",help="model path")
    parser.add_argument("--imgsz",type=int,default=224,help="test image size")
    #opt=parser.parse_known_args()[0]
    return parser

class Test_model():
    def __init__(self,opt):
        self.imgsz=opt.imgsz
        self.img_dir=opt.test_dir
        self.save_dir=opt.save_dir
        self.threshold=opt.threshold
        self.weight=opt.weights
        self.model=(torch.load(opt.weights)).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()
        self.class_name=['dust', 'good', 'mult', 'nucleus', 'part',  'suipian','vague']
        
    def __call__(self):
        data_transorform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.CenterCrop((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        img_list=glob.glob(self.img_dir+os.sep+"*.png")
        print(img_list[0])
        print(len(img_list))
        count=[0,0,0,0,0,0,0]
        null = 0
        destination_file = new_path = os.path.join(os.path.dirname(self.save_dir), os.path.basename(self.save_dir) + '.txt')
        with open( destination_file, 'w') as file:
            file.write(f"image path：{self.img_dir}\n")
            file.write(f"weight:{self.weight}\n")
            file.write(f"threshold:{self.threshold}\n")
            file.write("Image:\tpred:\tprob：\n")

            for imgpath in tqdm(img_list, desc="Processing Images"):
                img=cv2.imread(imgpath)
                h,w=img.shape[:2]
                if h<30 or w<30:continue
                new_img=self.expend_img(img)
                img=Image.fromarray(new_img)
                img=data_transorform(img)
                img=torch.reshape(img,(-1,3,self.imgsz,self.imgsz)).to(device)
                
                output = self.model(img)
                _, pred = torch.max(output.data, 1)
                max_value = output[0][pred].item()
                outputs = self.class_name[pred]
                os.makedirs(self.save_dir, exist_ok=True )
                    
                if outputs=="dust":
                    
                    root,dir = os.path.split(self.save_dir)
                    if max_value>=self.threshold:
                        count[0]+=1
                        continue
                        
                    elif output[0][1].item()>=0.3:
                        count[1]+=1
                        img_destination_directory = self.save_dir
                        os.makedirs(img_destination_directory, exist_ok=True )
                        img_save_name = f"{(output[0][1].item()):.4f}_"+"dust_"+basename(imgpath)
                        img_save_path = os.path.join(img_destination_directory, img_save_name)
                        shutil.copy(imgpath, img_save_path)
                        file.write(f"{basename(imgpath)}\t{outputs}\t{str(output[0].tolist())}\n")
                        
                    else:
                        count[0]+=1
                        continue
                    
                
                elif outputs=="good":
                    count[1]+=1
                    root,dir = os.path.split(self.save_dir)
                    if max_value>self.threshold:    
                        img_destination_directory = self.save_dir
                        os.makedirs(img_destination_directory, exist_ok=True )
                    
                    else :
                        img_destination_directory = self.save_dir
                        os.makedirs(img_destination_directory, exist_ok=True )
                    img_save_name = f"{(output[0][1].item()):.4f}_"+"good_"+basename(imgpath)
                    img_save_path = os.path.join(img_destination_directory, img_save_name)
                    shutil.copy(imgpath, img_save_path)
                    file.write(f"{basename(imgpath)}\t{outputs}\t{str(output[0].tolist())}\n")

                elif outputs=='mult':
                    root,dir = os.path.split(self.save_dir)
                    if max_value>=self.threshold:
                        count[2]+=1
                        continue
                        
                    elif output[0][1].item()>=0.3:
                        count[1]+=1
                        img_destination_directory =self.save_dir
                        os.makedirs(img_destination_directory, exist_ok=True )
                        img_save_name = f"{(output[0][1].item()):.4f}_"+"mult_"+basename(imgpath)
                        img_save_path = os.path.join(img_destination_directory, img_save_name)
                        shutil.copy(imgpath, img_save_path)
                        file.write(f"{basename(imgpath)}\t{outputs}\t{str(output[0].tolist())}\n")
                        
                    else:
                        count[2]+=1
                        continue
                    
                elif outputs=='nucleus':
                    count[3]+=1
                    continue
                elif outputs=='part':
                    count[4]+=1
                    continue
                
                elif outputs=='suipian':
                    count[5]+=1
                    continue

                elif outputs=='vague':
                    count[6]+=1
                    continue
                
            print(self.class_name)
            print(count)
            file.write(f"{self.class_name[0]}={count[0]}\t{self.class_name[1]}={count[1]}\t{self.class_name[2]}={count[2]}\n")
            file.write(f"{self.class_name[3]}={count[3]}\t{self.class_name[4]}={count[4]}\t{self.class_name[5]}={count[5]}\n")
            file.write(f"{self.class_name[6]}={count[6]}\n")

    def expend_img(self,img,fill_pix=255):
        fill_pix=[255,255,255]
        h,w=img.shape[:2]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb
    
if __name__ == '__main__':
    args = make_parser().parse_args()
    run_dir = args.test_dir
    file_name = os.path.basename(run_dir)
    
    save_dir = args.save_dir
    args.save_dir = save_dir
    weight = args.weights
    threshold = args.threshold

    print("save_dir:" + save_dir)
    print("weight:" + weight )
    print("threshold:" + str(threshold))
    os.makedirs(save_dir) 
    test_img=Test_model(args)
    test_img()
