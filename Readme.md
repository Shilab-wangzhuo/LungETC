# LungETC: An exfoliated tumor cell dataset annotated by single-cell sequencing for liquid biopsy-based lung cancer diagnosis and pathological subtyping

## Our Dataset

<div align="center"><img src=".\img\Figure1.png" width="550"></div>

**Figure1:** (A) Schematic illustration of BAL fluid specimen collection, Pap-stained BAL cytology, as well as expert-based ETC annotation and scDNA-Seq-based ETC annotation. (B) Pap-stained cytology images (bottom) and single-cell CNA profiles (top) of cells annotated by expert cytopathologists from BALF specimens.

**Download dataset:** [LungETC_dataset.zip](https://github.com/Shilab-wangzhuo/LungETC/raw/refs/heads/main/LungETC_dataset/LungETC_dataset.zip)


## Supplementary Information
### Extra Performance evaluation of binary classification tasks
<div align="center"><img src=".\img\FigureS1.png" width="750"></div>

**FigureS1:** (A) Confusion matrices of top four SOTA classifiers for malignant-benign cells classification. (B) Confusion matrices of top four SOTA classifiers for SCLC-NSCLC subtype classification. The darker the color, the greater the proportion of predictions for that category in the true labels. The optimal performance of the top four SOTA models on the test set is presented.

### Details of Single-cell Extraction and Segmentation
[Click Here for Details](./Readme_LungETC_preprocessing.md)


## Quick start
<div align="center"><img src=".\img\Figure2.png" width="750"></div>

**Figure2:** Overview of the pipeline using LungETC for BAL-based lung cancer diagnosis and pathological subtyping. (A) Single-cell extraction and segmentation workflow.(B) Malignant-benign classifier training.(C) Pathological subtyping model training.

Before you start, we recommend you to create a new conda environment. 

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install model framework

References:
- YOLOX-->[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- EfficientNet-->[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- UNet-->[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

```bash
cd tools/Step2_YOLOX/YOLOX
pip install -v -e .

tar -xf tools/Step4_qc/EfficientNet-PyTorch.zip -C tools/Step4_qc/EfficientNet-PyTorch
cd tools/Step4_qc/EfficientNet-PyTorch
pip install efficientnet_pytorch
```

## Usage

1. Download the pre-trained weights

- Download the yolox-x pre-trained model and put the folder in "tools\Step2_YOLOX\YOLOX\YOLOX_weights"

[https://1drv.ms/u/c/194ecbbe03b12717/EYqb1VjNxBNDjgWfqT__PP8BBf7OKnYgj2iFBgOiz-L56g?e=xbqGAk](https://1drv.ms/u/c/194ecbbe03b12717/EYqb1VjNxBNDjgWfqT__PP8BBf7OKnYgj2iFBgOiz-L56g?e=xbqGAk)


- Download the Unet pre-trained model and put the folder in "tools\Step5_cut\Pytorch-UNet-master\checkpoints"

[https://1drv.ms/u/c/194ecbbe03b12717/ES4G-bKiHttKsNPzAbeMXcEBngpmwenAAegtcBKlTzeJuw?e=O7Gdbx](https://1drv.ms/u/c/194ecbbe03b12717/ES4G-bKiHttKsNPzAbeMXcEBngpmwenAAegtcBKlTzeJuw?e=O7Gdbx)



3. Running the Process

- Step-by-step instructions

```bash
# Step 1: Split the whole slide images (WSIs) into non-overlapping patches of 1024×1024 pixels.
python tools/Step1_slide/svs_slide.py -i N1.svs -o /output_dir/Step1_slide 

# Step 2: Utilize the YOLOX network to detect single cells within the patch-level images.
python tools/Step2_YOLOX/YOLOX/tools/demo1.py image -n yolox-x -c tools/Step2_YOLOX/YOLOX/YOLOX_weights/best_ckpt.pth --path /output_dir/Step1_slide/N1 --save_dir /output_dir/Step2_YOLOX --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu

# Step 3: Extract single-cell images from the patch-level images based on the detection results of the YOLOX model.
python tools/Step3_sc_slide/sc_slide.py /output_dir/Step2_YOLOX/N1/output_dir/Step1_slide/N1/output_dir/Step3_sc_slide

# Step 4: Remove blurry cells, incomplete cells, cell fragments, multicellular clusters, impurities, and cell nuclei.
python tools/Step4_qc/QC.py --test_dir /output_dir/Step3_sc_slide/N1  --save_dir /output_dir/Step4_qc/N1 

# Step 5: Segment cells from the background in the single-cell images to effectively reduce background interference.
python tools/Step5_cut/Pytorch-UNet-master/predict.py -i /output_dir/Step4_qc/N1 -o  /output_dir/Step5_cut

# Step 6: Classify whether the cell is benign or malignant.
python tools/Step6_classify/efficient_classify.py --test_dir /output_dir/Step5_cut/N1/json_cut_out   --save_dir /output_dir/Step6_classify/N1 --ori_img_dir /output_dir/Step4_qc/N1 

```

- Run via Bash (Windows)
```bash
# before you start, make sure you activate your conda environment.
process.bat -i /your/input/svs_file/path -o /output/folder 
# [-o] is optional
```



## Demo

We present partial results of eight samples in the `demo` (including four positive samples and four negative samples).

The input file format and the output directory structure are shown as follows.
```  
Input file: N1.svs

Output directory structure:

    -- N1   
        - Step1_slide   
            - N1             # Non-overlapping patches of 1024×1024 pixels.
        - Step2_YOLOX
            - N1             # YOLOX model results.
        - Step3_sc_slide
            - N1             # Single-cell images.
        - Step4_qc
            - N1             # High-quality single-cell images.
            - N1.txt         # Probability values for each cell given by the QC model.
        - Step5_cut
            - N1
                - json               # JSON files for each single-cell image.
                - json_cut_out       # High-quality single-cell images with the background removed.
                - vis                # Masks of single-cell images.
        - Step6_classify   
            - N1
                - cancer             # Cell images identified as malignant (background removed).
                - cancer_ori         # Cell images identified as malignant (with backgroung).
            - N1.txt         # Malignancy probability values for each cell given by the binary classification model.
```

**Note: Due to the large number of image files, we only retained the results of five patch-level images in Steps 1 through 5. The results of Step 6 cover the entire WSI.**

## Process Illustration

1. Step 1: Image Preprocessing

Split the whole slide images (WSIs) into non-overlapping patches of 1024×1024 pixels.

2. Step 2: Single-cell Identification

Utilize the YOLOX network to detect single cells within the patch-level images.

3. Step 3: Single-cell Extraction

Extract single-cell images from the patch-level images based on the detection results of the YOLOX model.

4. Step 4: Quality Control

Remove blurry cells, incomplete cells, cell fragments, multicellular clusters, impurities, and cell nuclei.

5. Step 5: Single-cell Segmentation

Segment cells from the background in the single-cell images to effectively reduce background interference.

6. Step 6: Binary Classification

Classify whether the cell is benign or malignant.