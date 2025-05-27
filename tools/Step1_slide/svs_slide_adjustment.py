import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 50).__str__()
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse

def make_parser():
    parser = argparse.ArgumentParser("SVS slide!")
    
    parser.add_argument("-i", "--input_file", 
                        type=str, 
                        default=None,
                        help="please input your svs file")
    
    parser.add_argument("-s","--size", default=1024, type=int, help="output img size")
    parser.add_argument("--overlap", default=0.05, type=float, help="overlap rate of slide")
    parser.add_argument("-o", "--output_file", 
                        type=str, 
                        default=None,
                        help="please input your output dictionary")
    return parser

def process_svs_files(run_path):
    Info_list = []
    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file.endswith(".svs"): 
                svs_file_path = os.path.join(root, file)  
                print("svs_file_path:" + svs_file_path)
                svs_path, svs_name = os.path.split(svs_file_path)
                sample_name = os.path.splitext(svs_name)[0]
                file_dict = {'name': sample_name, 'svs_path': svs_file_path}
                Info_list.append(file_dict)
    return Info_list

# 新增：图像调整函数
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

def split_large_image(input_image_path, output_folder, tile_size, overlap_rate, row, col):
    slide = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    height, width = slide.shape[:2]
    overlap_pixels = int(tile_size * overlap_rate)

    left = col * (tile_size - overlap_pixels)
    top = row * (tile_size - overlap_pixels)
    right = min(left + tile_size, width)
    bottom = min(top + tile_size, height)
    patch = slide[top:bottom, left:right]
    
    # 应用图像调整
    patch = adjust_image(patch, gamma=1.1, brightness=1.1, contrast=1.01, r_gain=1.04, g_gain=1.02, b_gain=1.02)

    output_filename = f'tile_{row}_{col}.png'
    output_path = os.path.join(output_folder, output_filename)

    cv2.imwrite(output_path, patch)

def process_row(row, cols, input_image_path, output_folder, tile_size, overlap_rate):
    for col in range(cols):
        split_large_image(input_image_path, output_folder, tile_size, overlap_rate, row, col)

def split_large_image_parallel(input_image_path, output_folder, tile_size, overlap_rate, chunk_size=100):
    
    slide = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_LOAD_GDAL)
    height, width = slide.shape[:2]
    overlap_pixels = int(tile_size * overlap_rate)

    rows = (height - overlap_pixels) // (tile_size - overlap_pixels)
    cols = (width - overlap_pixels) // (tile_size - overlap_pixels)

    os.makedirs(output_folder, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = []
        for chunk_row in range(0, rows, chunk_size):
            for chunk_col in range(0, cols, chunk_size):
                chunk_rows = min(chunk_size, rows - chunk_row)
                chunk_cols = min(chunk_size, cols - chunk_col)
                futures.append(executor.submit(process_chunk, slide, output_folder, tile_size, overlap_rate, chunk_row, chunk_col, chunk_rows, chunk_cols))

        total_chunks = len(futures)
        with tqdm(total=total_chunks, desc="Processing", unit="chunks") as pbar:
            for future in futures:
                future.result()
                pbar.update(1)

def process_chunk(slide, output_folder, tile_size, overlap_rate, chunk_row, chunk_col, chunk_rows, chunk_cols):
    height, width = slide.shape[:2]
    overlap_pixels = int(tile_size * overlap_rate)

    for row in range(chunk_row, chunk_row + chunk_rows):
        for col in range(chunk_col, chunk_col + chunk_cols):
            left = col * (tile_size - overlap_pixels)
            top = row * (tile_size - overlap_pixels)
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            patch = slide[top:bottom, left:right]
            
            # 应用图像调整
            patch = adjust_image(patch, gamma=1.1, brightness=1.1, contrast=1.01, r_gain=1.04, g_gain=1.02, b_gain=1.02)

            output_filename = f'tile_{row}_{col}.png'
            output_path = os.path.join(output_folder, output_filename)

            cv2.imwrite(output_path, patch)

if __name__ == "__main__":
    args = make_parser().parse_args()
    if os.path.isdir(args.input_file):
        Info_list = process_svs_files(args.input_file)
    else:
        Info_list = []
        svs_file_path = args.input_file
        svs_path, svs_name = os.path.split(svs_file_path)
        sample_name = os.path.splitext(svs_name)[0]
        file_dict = {'name': sample_name, 'svs_path': svs_file_path}
        Info_list.append(file_dict)

    for sample in Info_list:
        input_image_path = sample['svs_path']
        print(input_image_path)
        sample_name = sample['name']
        print(sample_name)
        output_folder = os.path.join(args.output_file, sample_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(output_folder)
        tile_size = args.size
        overlap_rate = args.overlap
        split_large_image_parallel(input_image_path, output_folder, tile_size, overlap_rate)