from networks.detector import HeadDetector
from utils.functions import detect,get_file_id,draw_boxes
from config import cfg
import os
import torch
import cv2
import argparse

THRESH = 0.07
WIDTH, HEITH = 1920,1080
OVIDEO_PATH = '/content/drive/My Drive/DATN/FCHD/videos'
IVIDEO_PATH = '/content/drive/My Drive/DATN/FCHD/videos/entrance_camera_cutted_1.mp4'

if __name__ == "__main__":
  
    # Load model
    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    model_dict = torch.load(cfg.BEST_MODEL_PATH)['model']
    head_detector.load_state_dict(model_dict)
    head_detector = head_detector.cuda()
    
    # Read video
    vid = cv2.VideoCapture(IVIDEO_PATH)  
    frame_count = 0

    # Write video
    file_id=get_file_id(IVIDEO_PATH)
    out = cv2.VideoWriter(os.path.join(OVIDEO_PATH, file_id+'_result.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (WIDTH, HEITH))
    while(vid.isOpened()):
      _, frame = vid.read()
      if frame is None:
          break
      # if frame_count%5 == 0:
      bboxes,scores=detect(frame,head_detector,thresh=THRESH)
      frame=draw_boxes(frame,bboxes,scores)

      out.write(frame)
      frame_count += 1
    vid.release()
    out.release()
