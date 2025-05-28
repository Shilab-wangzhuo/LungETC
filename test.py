#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import datetime

def setup_logging(log_file):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directory_structure(output_dir):
    """创建输出目录结构"""
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建各步骤子目录
    step_folders = [
        "Step1_slide", "Step2_YOLOX", "Step3_sc_slide", 
        "Step4_qc", "Step5_cut", "Step6_malignant_benign_classification",
        "Step7_SCLC_NSCLC_classification"
    ]
    
    for folder in step_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    return True

def run_step(cmd, logger):
    """执行命令并记录输出"""
    logger.info(f"执行命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        logger.info(f"命令执行成功: {cmd}")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {cmd}")
        logger.error(f"错误信息: {e.stderr}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理图像的七步流程")
    parser.add_argument("-i", "--input_file", required=True, help="输入文件路径")
    parser.add_argument("-o", "--output_dir", help="输出目录路径")
    args = parser.parse_args()
    
    # 获取输入文件的绝对路径
    input_file = os.path.abspath(args.input_file)
    
    # 如果未指定输出目录，则使用输入文件所在目录
    if not args.output_dir:
        output_dir = os.path.dirname(input_file)
        output_dir = os.path.join(output_dir, Path(input_file).stem)
    else:
        output_dir = args.output_dir
    
    # 获取样本名称
    sample_name = Path(input_file).stem
    
    # 创建目录结构
    create_directory_structure(output_dir)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, "log_file.txt")
    logger = setup_logging(log_file)
    
    logger.info(f"开始处理样本: {sample_name}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    
    # 执行七个步骤
    steps = [
        # Step 1: 处理幻灯片
        f'python tools\\Step1_slide\\svs_slide.py -i "{input_file}" -o "{os.path.join(output_dir, "Step1_slide")}"',
        
        # Step 2: YOLOX处理
        f'python tools\\Step2_YOLOX\\YOLOX\\tools\\demo1.py image -n yolox-x -c tools\\Step2_YOLOX\\YOLOX\\YOLOX_weights\\best_ckpt.pth --path "{os.path.join(output_dir, "Step1_slide", sample_name)}" --save_dir "{os.path.join(output_dir, "Step2_YOLOX")}" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu',
        
        # Step 3: 幻灯片SC处理
        f'python tools\\Step3_sc_slide\\sc_slide.py "{os.path.join(output_dir, "Step2_YOLOX", sample_name)}" "{os.path.join(output_dir, "Step1_slide", sample_name)}" "{os.path.join(output_dir, "Step3_sc_slide")}"',
        
        # Step 4: 质量控制
        f'python tools\\Step4_qc\\QC.py --test_dir "{os.path.join(output_dir, "Step3_sc_slide", sample_name)}" --save_dir "{os.path.join(output_dir, "Step4_qc", sample_name)}"',
        
        # Step 5: 切割
        f'python tools\\Step5_cut\\Pytorch-UNet-master\\predict.py -i "{os.path.join(output_dir, "Step4_qc", sample_name)}" -o "{os.path.join(output_dir, "Step5_cut")}"',
        
        # Step 6: 良恶性分类
        f'python tools\\Step6_malignant_benign_classification\\malignant_benign_classfier.py --test_dir "{os.path.join(output_dir, "Step5_cut", sample_name, "json_cut_out")}" --save_dir "{os.path.join(output_dir, "Step6_malignant_benign_classification", sample_name)}" --ori_img_dir "{os.path.join(output_dir, "Step4_qc", sample_name)}"',
        
        # Step 7: SCLC/NSCLC分类
        f'python tools\\Step7_SCLC_NSCLC_classification\\SCLC_NSCLC_classifier.py --test_dir "{os.path.join(output_dir, "Step6_malignant_benign_classification", sample_name, "malignant")}" --save_dir "{os.path.join(output_dir, "Step7_SCLC_NSCLC_classification", sample_name)}" --ori_img_dir "{os.path.join(output_dir, "Step6_malignant_benign_classification", sample_name, "malignant_ori")}"'
    ]
    
    # 依次执行每个步骤
    for i, step_cmd in enumerate(steps, 1):
        logger.info(f"开始执行步骤 {i}/7")
        success = run_step(step_cmd, logger)
        if not success:
            logger.error(f"步骤 {i} 执行失败，流程终止")
            return 1
        logger.info(f"步骤 {i}/7 执行完成")
    
    logger.info(f"所有步骤执行完成，处理结果保存在: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())