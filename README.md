# detect-blood-cell-type-with-yolo
Finetuning yolo model to detect blood cell type

YOLOv7 is the fastest and most accurate real-time object detection model for computer vision tasks. In this repo, I've trained an object detection model on a custom dataset to detect RBC, WBC and PLATELETS using YOLOv7.

YOLOv7 repository: https://github.com/WongKinYiu/yolov7
```
git clone https://github.com/WongKinYiu/yolov7.git
```

YOLOv7 pretained weights: https://github.com/wongkinyiu/yolov7/releases  
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```

Robflow data (labeleld images) used for training: https://public.roboflow.com/object-detection/bccd 

# steps to follow

## Setup environment 

```
conda create -n yolov7 python=3.9
conda activate yolov7
pip install -r environment.txt
```

## Training of model 

```
python train.py --workers 8 --device 0 --batch-size 32 --data data/train.yaml --img 640 640 --cfg data/configuration.yaml --weights data/weights.pt  --name model --hyp data/hyper_parameters.yaml
```

## Testing of model 

```
python test.py --data data/test.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name model
```
