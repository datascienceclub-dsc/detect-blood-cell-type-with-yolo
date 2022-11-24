# detect-blood-cell-type-with-yolo
Finetuning yolo model to detect blood cell type

YOLOv7 is the fastest and most accurate real-time object detection model for computer vision tasks. In this repo, I've trained an object detection model on a custom dataset to detect RBC, WBC and PLATELETS using YOLOv7.

# steps to follow

## downloading yolo(v7) model
The first  step is to clone the YOLOv7 repository so that we can access the codebase for training the models.
Go to the folder and clone this : git clone https://github.com/WongKinYiu/yolov7.git       
Download pretrained weight : you can manually download  any version of the weights  from here, https://github.com/wongkinyiu/yolov7/releases   and then put the file in the yolov7 folder Or easily download it with this line of code : wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
Create an environment and activate the environment.
To create an environment with a specific version of Python: conda create -n yolov7 python=3.9
To activate the environment :  conda activate yolov7
To install modeules : pip install -r requirements.txt     


## downloading labelled images to be trained
For this object detection, I have used Robflow data. You can download it from here.
https://public.roboflow.com/object-detection/bccd

## Training of model 
python train.py --workers 8 --device 0 --batch-size 32 --data data/train.yaml --img 640 640 --cfg data/configuration.yaml --weights data/weights.pt  --name model --hyp data/hyper_parameters.yaml

python test.py --data data/test.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name model