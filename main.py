import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import time
import torch

sys.path.insert(0, os.getcwd())

from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from yolox.tracker.byte_tracker import BYTETracker

ckpt = torch.load(
    '/content/LEAF-YOLO/cfg/LEAF-YOLO/leaf-sizes/weights/best.pt',
    map_location='cuda',
    weights_only=False
)
print(ckpt.keys())

frame_dir = "/content/drive/MyDrive/Data/VisDrone2019-MOT-val/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
gt_file_path = "/content/drive/MyDrive/Data/VisDrone2019-MOT-val/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt"

class TrackerArgs:
    track_thresh = 0.50
    track_buffer = 80
    match_thresh = 0.80
    mot20 = False

tracker = BYTETracker(
    TrackerArgs(),
    frame_rate=30
)

def calculate_gmc(prev_gray, curr_gray):
    if prev_gray is None or curr_gray is None:
        return 0.0, 0.0
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=20, blockSize=3)
    if prev_pts is None:
        return 0.0, 0.0
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 4:
        return 0.0, 0.0
    matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
    if matrix is None:
        return 0.0, 0.0
    return matrix[0, 2], matrix[1, 2]

frame_files = sorted(os.listdir(frame_dir))
frames = []

for file in frame_files:
    frame = cv2.imread(os.path.join(frame_dir, file))
    frames.append(frame)

batch_size = 16 

detections = []

model.eval()

start_det = time.time()

for i in range(0, len(frames), batch_size):
    batch_frames = frames[i:i+batch_size]
    batch_tensor = []

    for frame in batch_frames:
        test = letterbox(frame, 640, stride=32)[0]
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        test = np.ascontiguousarray(test)
        test = torch.from_numpy(test).float() / 255
        batch_tensor.append(test)

    batch_tensor = torch.stack(batch_tensor).cuda()

    with torch.no_grad():
        predictions = model(batch_tensor)[0]

    filtered_predictions = non_max_suppression(
        predictions,
        conf_thres=0.15,
        iou_thres=0.70
    )

    for frame, det in zip(batch_frames, filtered_predictions):
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                batch_tensor.shape[2:],
                det[:, :4],
                frame.shape
            ).round()
            det = det.cpu().numpy()
        else:
            det = np.empty((0, 6))

        detections.append(det)

end_det = time.time()