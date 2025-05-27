#!/bin/bash

# 初始化输入目录和输出目录的变量
input_dir=""
output_base_dir=""

# 显示使用方法
usage() {
    echo "Usage: $0 -i <input_directory> [-o <output_base_directory>]"
    exit 1
}

# 解析命令行参数
while getopts "i:o:" opt; do
    case $opt in
        i) input_dir="$OPTARG";;
        o) output_base_dir="$OPTARG";;
        ?) usage;;
    esac
done

# 检查输入目录参数
if [ -z "$input_dir" ]; then
    usage
fi

# 转换为绝对路径
input_dir=$(realpath "$input_dir")

# 如果未提供输出基础目录，则默认为输入目录
if [ -z "$output_base_dir" ]; then
    output_base_dir="$input_dir"
fi

# 确保输出基础目录存在
mkdir -p "$output_base_dir"

# 创建日志目录
log_dir="$output_base_dir/logs"
mkdir -p "$log_dir"

# 处理单个文件的函数
process_file() {
    local input_file="$1"
    local name=$(basename "$input_file" .svs)
    local output_dir="$output_base_dir/${name}"
    local log_file="$log_dir/${name}_log.txt"

    echo "Processing file: $input_file"
    echo "Output directory: $output_dir"

    # 创建输出目录
    mkdir -p "$output_dir"

    # 创建所需的子目录
    for step in Step1_slide Step2_YOLOX Step3_sc_slide Step4_qc Step5_cut Step6_classify; do
        mkdir -p "$output_dir/$step"
    done

    # 记录开始时间
    echo "Started processing $name at $(date)" >> "$log_file"

    # 激活conda环境并执行处理步骤
    {
        # Step 1-4 使用 learn 环境
        echo "Activating learn environment..."
        source activate learn
        
        python tools/Step1_slide/svs_slide.py -i "$input_file" -o "$output_dir/Step1_slide"
        
        python tools/Step2_YOLOX/YOLOX/tools/demo1.py image -n yolox-x \
            -c tools/Step2_YOLOX/YOLOX/YOLOX_weights/best_ckpt.pth \
            --path "$output_dir/Step1_slide/$name" \
            --save_dir "$output_dir/Step2_YOLOX" \
            --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu
        
        python tools/Step3_sc_slide/sc_slide.py \
            "$output_dir/Step2_YOLOX/$name" \
            "$output_dir/Step1_slide/$name" \
            "$output_dir/Step3_sc_slide"
        
        python tools/Step4_qc/QC.py \
            --test_dir "$output_dir/Step3_sc_slide/$name" \
            --save_dir "$output_dir/Step4_qc/$name"

        # Step 5 使用 myenv 环境
        echo "Activating myenv environment..."
        source activate myenv
        
        python tools/Step5_cut/Pytorch-UNet-master/predict.py \
            -i "$output_dir/Step4_qc/$name" \
            -o "$output_dir/Step5_cut"

        # Step 6 回到 learn 环境
        echo "Activating learn environment..."
        source activate learn
        
        python tools/Step6_classify/efficient_classify.py \
            --test_dir "$output_dir/Step5_cut/$name/json_cut_out" \
            --save_dir "$output_dir/Step6_classify/$name" \
            --ori_img_dir "$output_dir/Step4_qc/$name"

    } >> "$log_file" 2>&1

    echo "Finished processing $name at $(date)" >> "$log_file"
}

# 主处理循环
echo "Starting batch processing..."
for svs_file in "$input_dir"/*.svs; do
    if [ -f "$svs_file" ]; then
        process_file "$svs_file"
    fi
done

echo "Batch processing completed!"