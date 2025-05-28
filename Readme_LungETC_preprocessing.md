# LungETC: Single-cell extraction and segmentation

<div align="center"><img src=".\img\Figure2.png" width="750"></div>
<!-- <div style="text-align: center; font-weight: bold;">Figure2</div>
Here we give more details about the Single-cell extration and segmentation step. -->

## Single-cell extraction

A YOLOX network was trained to detect single cells in patch-level images. During inference, patch-level images were input, and annotations in ‘json’ format were output. Regions with a confidence score > 0.3 and IoU > 0.5 (after NMS) were retained, cropped, and used as input for the next stage.

<div align="center"><img src=".\img\FigureS2.png" width="750"></div>
<!-- <div style="text-align: center; font-weight: bold;">FigureS2</div> -->


Reference: YOLOX-->[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

### YOLOX Training Details

 We trained the network on 219 patch-level images with 5496 annotated boxes, split into training and validationsets in a 9:1 ratio. The batch size was set to 64 and the model parameters were updated using the SGD optimizer with Nesterov Accelerated Gradient (momentum factor=0.9) and L2 weight decay was set to 0.0005. We adopted the ‘yoloxwarmcos’ learning rate scheduling strategy from the YOLOX open-source framework. We combineda warm-up phase of 5 epochs followed by cosine annealing. During the warm-up phase, the learning rate increased from 0 to 0.01 first, then gradually decreased to 0.0005 and remained at the same level.

### YOLOX Performance Evaluation
 In validation set of 21 patch-level images, our model achieved a **COCOAP50** score of **0.8625** and a **COCOAP50-95** score of **0.5064**.

 We also evaluated the model on a test set of 200 patch-level images, in which we manual annotated 4069 cells totally while The model annotated a total of 6,550. Among the model's annotations, 3,930 cells were correctly detected, 242 were duplicate detections, and 2,378 were low-quality detections. Compared to the manual annotations, there were 139 missed detections. Since the low-quality detections will be removed during the subsequent quality control step, we only consider the missed and duplicate detections as errors, totaling 381, which accounts for 9.4% of all manually annotated single cells.

<div align="center"><img src=".\img\FigureS3.png" width="550"></div>
<!-- <div style="text-align: center; font-weight: bold;">FigureS3</div> -->

## Single-cell Quality Control

After single-cell extraction, a QC model was used to remove non-cellular constituents (e.g., nuclei, debris, incomplete cells). A dataset of low-quality images with 6 categories was created, and single-cell images were resized to 224×224 pixels. A seven-class classification model, based on a pre-trained EfficientNet-B2, was trained to filter out low-quality images.

<div align="center"><img src=".\img\FigureS4.png" width="750"></div>
<!-- <div style="text-align: center; font-weight: bold;">FigureS4</div> -->

Reference: EfficientNet-->[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

### QC Model Training Details

 The fully connected layer of the original EfficientNet-B2 network was replaced with a sequential module, including a linear layer and a softmax activation function. The model was fine-tuned using 757 high-quality and 2,071 low-quality single-cell images, split into training and validation sets in a 7:3 ratio. An SGD optimizer with an L2 weight decay of 0.0004, a momentum of 0.96, and a batch size of 8 was used. The learning rate increased linearly from 0.00009 to 0.0004, sustained for 5 epochs, and then decayed exponentially with a factor of 0.93 over 50 epochs. High-quality images from this stage were used for downstream analysis.

### QC Model Performance Evaluation
Our QC model achieved an accuracy of 0.7889 on the validation set. We further evaluated its performance on the test set used in the YOLOX model. After processing with the quality control model, **the percentage of low-quality objects significantly decreased from 37.7±18.6% to 3.1±5.8%, while the original cell recovery rate of 94.3±6.9% was still maintained at 84.2±11.5%**.


## Single-cell Segmeantation
<div align="center"><img src=".\img\FigureS5.png" width="750"></div>
<!-- <div style="text-align: center; font-weight: bold;">FigureS5</div> -->

Reference: UNet-->[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)


### UNet Training Details
A U-Net model was trained to segment cells from the background in single-cell images, reducing background interference. A total of 1,423 single-cell images were annotated using AnyLabeling (v0.3.3) to generate labeled ‘json’ files, which were converted into mask images in ‘png’ format and split into training and validation sets at a 9:1 ratio. The U-Net model, consisting of 19 CNN blocks, was trained for 20 epochs using the RMSprop optimizer with a learning rate of 2×10⁻⁵, a weight decay of 1×10⁻⁸, a momentum of 0.95, and batch sizes of 4. The loss function combined PyTorch’s ‘BCEWithLogitsLoss’ and Dice loss to optimize the Dice score. During inference, input images were resized to 224×224 pixels, and only predicted regions containing single cells were retained for analysis.


### UNet Performance Evaluation
Our U-Net model achieved **a Dice score of 0.92935** on the validation set, indicating high segmentation accuracy.