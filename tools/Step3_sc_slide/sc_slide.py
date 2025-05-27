import os
import json
from PIL import Image
import sys

def process_png_files(json_path,img_path):
    Info_list = []
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                json_path, json_name = os.path.split(json_file_path)
                png_file_path = os.path.join(img_path, json_name.replace(".json", ".png"))
                dir_name = os.path.basename(os.path.dirname(png_file_path))
                
                
                file_dict = {'name': json_name, 'png_path': png_file_path, 'json_path': json_file_path,'dir_name': dir_name}
                print(file_dict)
                Info_list.append(file_dict)
    return Info_list

def crop_images_from_json(json_file, png_path, png_name, dir_name,output_folder):
    img = Image.open(png_path)
    w,h = img.size
    with open(json_file, 'r', encoding='utf-8') as json_file:
        annotations = json.load(json_file)

    folder_path = os.path.join(output_folder, dir_name)
    
    os.makedirs(folder_path, exist_ok=True)

    for annotation in annotations["bboxes"]:
        if annotation.get("label"):
            x_min = int(annotation["x_min"] * w)
            x_max = int(annotation["x_max"] * w)
            y_min = int(annotation["y_min"] * h)
            y_max = int(annotation["y_max"] * h)

            patch = img.crop((x_min, y_min, x_max, y_max))

            final_name = png_name.replace(".json", "_") + str(annotation['id']) + ".png"
            save_path = os.path.join(folder_path, final_name)
            patch.save(save_path)

            print(f"保存图像：{save_path}")

if __name__ == "__main__":
    
    json_path = sys.argv[1]
    img_path = sys.argv[2]
    output_folder = sys.argv[3]
    
    os.makedirs(output_folder, exist_ok=True)

    Info_list =  process_png_files(json_path,img_path)

    for sample in Info_list:
        json_path = sample['json_path']
        png_path = sample['png_path']
        png_name = sample['name']
        dir_name = sample['dir_name']
        crop_images_from_json(json_path, png_path, png_name, dir_name, output_folder)
        
# command e.g
# python sc_slide.py 
# yolox/home/YOLOX/YOLOX_outputs/yolox_x/vis_res/2024_03_04_18_17_20 
# svs_slide_output1/2023-09-01_17_29_56  
# sc_slide_output