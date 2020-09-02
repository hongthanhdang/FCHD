from skimage import transform
from data.preprocess import Normalize
import os
import torch
import numpy as np
import time
import cv2
import math

def preprocess(img, min_size=600, max_size=1000):
    c, h, w = img.shape
    scale = min(min_size / min(h, w), max_size / max(h, w))
    # img = img / 255
    img = transform.resize(img, (c, h * scale, w * scale), mode='reflect') # rescale image to 600x800
    # print(img.shape)
    normalize = Normalize(mode='caffe')
    sample = normalize({'img': img}) # image size 600 x 800
    return sample['img'], scale

def read_img(img):
    img_raw = img.copy()
    img = img.transpose((2, 0, 1))  # (H,W,C) -> (C,H,W)
    img, scale = preprocess(img)
    return img, img_raw, scale

def detect(img, head_detector,thresh):
    # Pre-process img
    img, img_raw, scale = read_img(img)
    img = np.expand_dims(img, axis=0)
    # print("Image shape: ",img.shape)
    # print("Scale: ",scale)
    img = torch.from_numpy(img)
    img = img.cuda().float()

    # Inference
    begin = time.time()
    preds, scores = head_detector(img, scale, score_thresh=thresh)
    end = time.time()
    # print("[INFO] Model inference time: {:.3f} s".format(end - begin))
    # print("[PREDS SCORES]\n", scores)
    # print("[PREDS SCORES]\n", scores)
    # print("[INFO] raw image size: ",img_raw.shape)
    # print("[INFO] image size:",img.shape)
    
    # Rescale boxes
    bboxes=rescale_boxes(preds,scale)
    return bboxes, scores
    

def draw_boxes(img_raw,preds,scores):
  for bbox,score in zip(preds,scores):
        ymin, xmin, ymax, xmax = bbox
        cv2.rectangle(img_raw, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(img_raw, '{:.3f}'.format(score), (xmin, ymin - 9), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), thickness=1, lineType=8)
  return img_raw

def rescale_boxes(preds,scale):
  bboxes=[]
  for bbox in preds:
    ymin, xmin, ymax, xmax = bbox
    xmin, ymin = int(xmin / scale), int(ymin / scale)
    xmax, ymax = int(xmax / scale), int(ymax / scale)
    bboxes.append([ymin,xmin,ymax,xmax])
  return bboxes

def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def is_old (center_Xd,center_Yd,tracking_boxes, max_distance):
    for box_tracker in tracking_boxes:
        (xt, yt, wt, ht) = [int(c) for c in box_tracker]
        center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
        distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)
        if distance < max_distance:
            return True
    return False

def get_box_info(predicted_box ):
    # print(predicted_box)
    (ymin, xmin, ymax, xmax) = [int(v) for v in predicted_box]
    center_X = int((xmin + xmax) / 2.0)
    center_Y = int((ymin + ymax) / 2.0)
    w=int(xmax-xmin)
    h=int(ymax-ymin)
    return xmin, ymin, w, h, center_X, center_Y

def get_box_detail(tracked_box):
    (x, y, w, h) = [int(v) for v in tracked_box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y

