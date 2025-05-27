import argparse
import logging
import os
import glob
from PIL import Image
import json
import cv2
from labelme import utils
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms


from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from shapely.geometry import Polygon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_from_polygon(image_path,image_shape, polygon):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255  # start with white background
    mask_colour = np.ones((image_shape[0],image_shape[1],3), dtype=np.uint8)
    try:
        if polygon is None:
            raise ValueError("largest_polygon is None. Ensure it's properly initialized before calling mask_from_polygon.")
        if not isinstance(polygon, (list, np.ndarray)) or len(polygon) == 0:
            raise ValueError("largest_polygon should be a non-empty list or numpy array of points.")
        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 0)  # fill the polygon with black
        cv2.fillPoly(mask_colour, polygon, [0, 255, 0])  # fill the polygon with black
        result = image.copy()
        result[mask == 255] = (255, 255, 255) 
        alpha = 0.8  # 标签图像的透明度
        overlayed_image = cv2.addWeighted(image, alpha, mask_colour, 1 - alpha, 0)
    
    # overlayed_image = Image.fromarray(overlayed_image)
        return result,overlayed_image
    except TypeError as e:
        print(f"TypeError encountered: {e}")
        print(f"Polygon type: {type(polygon)}")
        print(f"Polygon value: {polygon}")
        return "none","none"
    



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    largest_polygon = None
    polygons = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour) > 2:
            points = contour.tolist()
            polygon = Polygon(points)
            area = polygon.area
            if area > max_area:
                max_area = area
                largest_polygon = points
    polygons.append(largest_polygon)
    return polygons

def mask_to_labelme_json(mask, filename,temp_json_filename,temp_json_cut_out_filename,temp_vis_filename,crop_para):
    
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    out = Image.fromarray(out)
    if crop_para[0][0]>224 or  crop_para[0][1]>224:
        max_value = max(crop_para[0][0],crop_para[0][1])
        out_scale = out.resize((max_value,max_value))
        out_crop = out_scale.crop(crop_para[1])
        out_final = out_crop
    else:
        out_crop = out.crop(crop_para[1])
        # out_scale = out_crop.resize(crop_para[0])
        out_final = out_crop
    
    # out_scale = out_scale.convert('L')
    out_final = np.array(out_final)

    largest_polygon = mask_to_polygon(out_final)
    
    data = {
        "version": "4.5.9",  # LabelMe version
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(filename),
        "imageData": None,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1],
    }

    
    shape = {
        "label": "foreground",
        "points": largest_polygon,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }
    data["shapes"].append(shape)

    # image_data = utils.img_arr_to_b64(np.array(Image.open(image_path)))
    # data["imageData"] = image_data.decode('utf-8')

    with open(temp_json_filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    # cv2.imwrite(temp_json_cut_out_filename, result_image)
    # print(f"Saved cut-out image to {temp_json_cut_out_filename}")
    if largest_polygon==[None]:
        print("largest_polygon is None, skipping mask_from_polygon.")
        return
    if not isinstance(largest_polygon, (list, np.ndarray)) or len(largest_polygon) == 0:
        print("largest_polygon is invalid, skipping mask_from_polygon.")
        return
    result_image,overlayed_image = mask_from_polygon(filename,crop_para[0], largest_polygon)
    if not isinstance(result_image, np.ndarray) or not isinstance(overlayed_image, np.ndarray):
        return
    cv2.imwrite(temp_vis_filename, overlayed_image)
    cv2.imwrite(temp_json_cut_out_filename, result_image)
    # print(f"Saved cut-out image to {temp_json_cut_out_filename}")

    

    # Save the result image
    


def load_image1(img):
        # image = cv2.imread(path)
    image = Image.open(img).convert("RGB")
    original_image = np.array(image)
    original_height, original_width, channels = original_image.shape

    # 定义目标尺寸
    if original_height <= 224 and original_width <=224:
        target_height, target_width = 224, 224
    elif original_height > 224 and original_height >= original_width:
        target_height, target_width = original_height, original_height
    elif original_width > 224  and original_width > original_height:
        target_height, target_width = original_width, original_width

    # 创建一个全白图像
    white_image = np.ones((target_height, target_width,channels), dtype=np.uint8) * 255

    # 计算原始图像在目标图像中的起始位置
    start_y = (target_height - original_height) // 2
    start_x = (target_width - original_width) // 2

    # 确定原始图像的结束位置
    end_y = start_y + original_height
    end_x = start_x + original_width

    white_image[start_y:end_y, start_x:end_x] = original_image
    target_size = (224, 224)
    result_image = Image.fromarray(white_image)
    result_image = result_image.resize(target_size)
    crop_para = [(original_height,original_width),(start_x,start_y,end_x,end_y)]
    return result_image,crop_para

def get_args():
    
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='tools/Step5_cut/Pytorch-UNet-master/checkpoints/checkpoint_epoch12.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values,crop_para):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    out = Image.fromarray(out)

    return out

def cut_image(img, mask: np.ndarray, mask_values):
    if mask_values == [0, 1]:
        out = np.array(img)
        out[mask == 0] = [255,255,255]
        return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    if len(in_files) == 1 and os.path.isdir(in_files[0]):
        if args.output is not None:
            if len(args.output) == 1 and os.path.isdir(args.output[0]):
                out_dir = os.path.join(args.output[0],os.path.basename(in_files[0]))
            else:
                raise ValueError("Please provide a valid directory path for output.")
        else:
            out_dir = os.path.join(os.path.dirname(in_files[0]), os.path.basename(in_files[0])+"_cut")
        
        os.makedirs(out_dir,exist_ok=True)
        # cut_out_path = os.path.join(out_dir,"output")
        # os.makedirs(cut_out_path,exist_ok=True)
        # temp_img_path = os.path.join(out_dir,"temp","img")
        # os.makedirs(temp_img_path,exist_ok=True)
        # temp_mask_path = os.path.join(out_dir,"temp","mask")
        # os.makedirs(temp_mask_path,exist_ok=True)
        temp_json_path = os.path.join(out_dir,"json")
        os.makedirs(temp_json_path,exist_ok=True)
        temp_vis_path = os.path.join(out_dir,"vis")
        os.makedirs(temp_vis_path,exist_ok=True)
        temp_json_cut_path = os.path.join(out_dir,"json_cut_out")
        os.makedirs(temp_json_cut_path,exist_ok=True)

        img_list=glob.glob(in_files[0]+os.sep+"*.png")
        for filename in tqdm(img_list,desc="Processing Images"):
            # logging.info(f'Predicting image {filename} ...')
            img_ori =  Image.open(filename).convert("RGB")
            # img = load_image(filename)
            img,crop_para = load_image1(filename)

            mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            
            png_name = os.path.basename(filename)
            # cut_out_filename = os.path.join(cut_out_path,png_name)
            # cut_result = cut_image(img,mask, mask_values)
            # cut_result.save(cut_out_filename)
        
            # temp_img_filename = os.path.join(temp_img_path,png_name)
            # img.save(temp_img_filename)

            # temp_mask_filename = os.path.join(temp_mask_path,png_name)
            # temp_mask_result = mask_to_image(mask, mask_values,crop_para)
            # temp_mask_result.save(temp_mask_filename)

            temp_json_filename = os.path.join(temp_json_path,os.path.splitext(png_name)[0]+".json")
            
            # temp_mask_result.save(temp_mask_filename)
            temp_vis_filename = os.path.join(temp_vis_path,png_name)

            temp_json_cut_out_filename = os.path.join(temp_json_cut_path,png_name)
            mask_to_labelme_json(mask,filename, temp_json_filename,temp_json_cut_out_filename,temp_vis_filename,crop_para)
            # process_image_and_json(temp_img_filename, temp_json_filename, temp_json_cut_out_filename)
            # logging.info(f'Mask saved to {temp_json_cut_out_filename}')


    else:
        out_files = get_output_filenames(args)
        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            img = load_image1(filename)

            mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)
