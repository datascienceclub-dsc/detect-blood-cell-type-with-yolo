import os
import sys
from numpy import random
from pathlib import Path
import cv2
import torch
from tkinter import filedialog

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, scale_coords, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box

from datetime import datetime

# Configuration
#model_path = r'D:\AI-Developments\YOLOV7\yolov7-main\runs\train\bath+bed_6tags_300epochs\weights\best.pt'
#model_path = r'D:\AI-Developments\YOLOV7\yolov7-main\runs\train\bath+bed+living_12tags_300epochs\weights\best.pt'
#model_path = r'D:\AI-Developments\YOLOV7\yolov7-main\runs\train\yolov7_2022101618\weights\OD_0-10-16-01.pt'
#model_path=r'E:\YOLO\runs\train\Model_300\weights\best.pt'
model_path=r'D:\github_projects\detect-blood-cell-type-with-yolo\runs\train\Model_300\weights\best.pt'
save_path = r'D:\github_projects\detect-blood-cell-type-with-yolo\output'
#save_path = r'E:\YOLO\output'

imgsz = 640
weights = model_path

opt = {'device': 'cpu', 'img_size': 640, 'conf_thres': 0.30, 'iou_thres': 0.45, 'classes': None, 'agnostic_nms': False, 'augment': False}

save_img = True
view_img = False

# Initialize
#set_logging()
device = select_device(opt['device'])
half = device.type != 'cpu'  # half precision only supported on CUDA

# Functions
def detectObjects(source):
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get object names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 130) for _ in range(3)] for _ in names]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #if img.size
        # Inference
        pred = model(img, augment=opt['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])

        # Process detections
        for i, det in enumerate(pred): # For Each Image
            im0 = im0s
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                print("")
                print("Results:")
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                    no_of_objects = int(n)
                    category_name = names[int(c)]

                    print(f"Found {no_of_objects} object{'s' if(no_of_objects>1) else ''} of {category_name}")
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                    
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(os.path.join(save_path, os.path.basename(source)), im0)
                        print(f"The image with the result is saved in: {os.path.join(save_path, os.path.basename(source))}")
                    # Open saved image
                    counter = 0
                    while not os.path.exists(os.path.join(save_path, os.path.basename(source))):
                        time.sleep(1)
                        counter += 1
                        if(counter > 5):
                            break
                    if(os.path.exists(os.path.join(save_path, os.path.basename(source)))):
                        os.system('start "" "'+os.path.join(save_path, os.path.basename(source)+'"'))
                    else:
                        print("Failed to open Class Activation Map Image ", os.path.join(save_path, os.path.basename(source)))
            else:
                print("Could not detect object")
    return True

# Load model
start_time = datetime.now()
model = attempt_load(weights, map_location=device)  # load FP32 model
end_time = datetime.now()
print(f"Model Loaded in {end_time-start_time}")

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

# Traced Model
start_time = datetime.now()
model = TracedModel(model, device, opt['img_size'])
end_time = datetime.now()
print(f"Model Traced in {end_time-start_time}")

try:
    counter = 0
    while(True):
        counter += 1
        image_path = filedialog.askopenfilename()
        starttime = datetime.now()
        if(image_path == ''):
            break
        detectObjects(image_path)
        endtime = datetime.now()
        print("Time difference:", endtime-starttime)
except KeyboardInterrupt:
    print("Program Stopped")