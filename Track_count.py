import sys
import os
sys.path.insert(0, './YOLOX')
import torch
import cv2
from yolox.utils import vis
import time
from yolox.exp import get_exp
import numpy as np
from collections import deque
from collections import Counter
from byte_tracker import BYTETracker
# importing Detector
from yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Predictor

# Importing Deepsort
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Importing Visuals
from visuals import *

from intersect_ import *

# A Dictionary to keep data of tracking
data_deque = {}

class_names = COCO_CLASSES


lines  = [
    {'Title' : 'Line1', 'Cords' : [(680,650), (950,450)]}
]

object_counter = {
    'Line1' : Counter()
}




#Draw the Lines
def draw_lines(lines, img):
    for line in lines:
        img = cv2.line(img, line['Cords'][0], line['Cords'][1], (0,255,0), 3)
    return img

# Update the Counter
def update_counter(centerpoints, obj_name):
    for line in lines:
        p1 = Point(*centerpoints[0])
        q1 = Point(*centerpoints[1])
        p2 = Point(*line['Cords'][0])
        q2 = Point(*line['Cords'][1])
        if doIntersect(p1, q1, p2, q2):
            object_counter[line['Title']].update([obj_name])
            return True
    return False

# Draw the Final Results
def draw_results(img):
    height, width, _ = img.shape 
    x = width - 300
    y = 100
    offset = 0
    for line_name, line_counter in object_counter.items():
        Text = line_name + " : " + ' '.join([f"{label}={count}" for label, count in line_counter.items()])
        cv2.putText(img, Text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        y = y+offset
    return img



# Function to calculate delta time for FPS when using cuda
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# Draw the boxes having tracking indentities 
def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape 
    # Cleaning any previous Enteries
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) +offset[0]  for i in box]  
        box_height = (y2-y1)
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in set(data_deque):  
          data_deque[id] = deque(maxlen= 100)

        color = compute_color_for_labels(object_id[i])
        obj_name = class_names[object_id[i]]
        label = '%s' % (obj_name)
        
        data_deque[id].appendleft(center) #appending left to speed up the check we will check the latest map
        UI_box(box, img, label=label + str(id), color=color, line_thickness=3, boundingbox=True)

        if len(data_deque[id]) >=2:
            update_counter(centerpoints = data_deque[id], obj_name = obj_name)

    return img

# Tracking class to integrate Deepsort tracking with our detector
class Tracker():
    def __init__(self, filter_classes=None, model='yolox-s', ckpt='wieghts/yolox_s.pth'):
        self.detector = Predictor(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_classes = filter_classes
    def update(self, image, visual = True, logger_=True):
        height, width, _ = image.shape
        t1 = time_synchronized()
        _,info = self.detector.inference(image, visual=False, logger_=logger_)
        t2 = time_synchronized()
        outputs = []
        
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            objectids = []
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_classes:
                    if class_names[class_id] not in set(filter_classes):
                        continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])  
                objectids.append(info['class_ids'])             
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            t3 = time_synchronized()
            outputs = self.deepsort.update(bbox_xywh, scores, info['class_ids'],image)
            t4 = time_synchronized()

            data = []
            if len(outputs) > 0:
                if visual:
                    if len(outputs) > 0:
                        bbox_xyxy =outputs[:, :4]
                        identities =outputs[:, -2]
                        object_id =outputs[:, -1]
                        image = draw_boxes(image, bbox_xyxy, object_id,identities)
            return image, outputs, t4-t3,t2-t1


if __name__=='__main__':
    # torch.set_default_device('mps')
    filter_classes = ['person'] # We are only interested in person
    tracker = Tracker(filter_classes=None, model='yolox-s', ckpt='weights/yolox_s.pth')    # instantiate Tracker

    # video_path = sys.argv[1]
    video_path = './videos/retail.mp4'

    cap = cv2.VideoCapture(video_path) 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the directory and filename of the input video
    video_dir, video_filename = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_filename)

    # Modify the output video path to be in the same directory and with the desired name format
    output_video_path = os.path.join(video_dir, f'count_demo_{video_name}.mp4')
    
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )


    frame_count = 0
    fps = 0.0
    while True:
        ret_val, frame = cap.read() # read frame from video
        if ret_val:
            t1 = time_synchronized()
            frame, bbox, time_inf, time_tracker = tracker.update(frame, visual=True, logger_=False)  # feed one frame and get result
            t2 = time_synchronized()
            frame = draw_lines(lines, img = frame)
            frame = draw_results(img= frame)
            vid_writer.write(frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            fps  = ( fps + (1./(time_synchronized()-t1)) ) / 2
            frame_count += 1
            print(f'video ({frame_count}/{total_frames}) {video_path}: ({(1E3 * (t2 - t1)):.1f}ms) Total Inference ({1E3 * time_inf:.1f}) Inf , ({1E3 * time_tracker:.1f}) Tracker')
        else:
            break

    cap.release()
    vid_writer.release()
    # cv2.destroyAllWindows()
