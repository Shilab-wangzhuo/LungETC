@echo off
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401929 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401930 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401933 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401935 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401942 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202401944 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202402216 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202402222 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202402223 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]
python tools/demo1.py image -n yolox-x -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth --path F:\neg_train_dataset_svs\Step1\C202402227 --save_dir "F:\neg_train_dataset_svs\Step2" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device [gpu]

python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401929"  "F:\neg_train_dataset_svs\Step1\C202401929"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401930"  "F:\neg_train_dataset_svs\Step1\C202401930"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401933"  "F:\neg_train_dataset_svs\Step1\C202401933"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401935"  "F:\neg_train_dataset_svs\Step1\C202401935"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401942"  "F:\neg_train_dataset_svs\Step1\C202401942"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202401944"  "F:\neg_train_dataset_svs\Step1\C202401944"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202402216"  "F:\neg_train_dataset_svs\Step1\C202402216"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202402222"  "F:\neg_train_dataset_svs\Step1\C202402222"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202402223"  "F:\neg_train_dataset_svs\Step1\C202402223"  "F:\neg_train_dataset_svs\Step3"
python F:\process_code\Step3\sc_slide.py "F:\neg_train_dataset_svs\Step2\C202402227"  "F:\neg_train_dataset_svs\Step1\C202402227"  "F:\neg_train_dataset_svs\Step3"