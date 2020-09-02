from networks.detector import HeadDetector
from utils.functions import detect,get_file_id,draw_boxes
from config import cfg
import os
import torch
import cv2
import argparse

THRESH=0.01

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img_path", help="path of the input image")
    args = parser.parse_args()

    # Load model
    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    model_dict = torch.load(cfg.NEW_MODEL_PATH)['model']
    head_detector.load_state_dict(model_dict)
    head_detector = head_detector.cuda()

    # Read image
    img = cv2.imread(args.img_path)
    
    # Predict and draw boxes
    bboxes,scores= detect(img,head_detector,thresh=THRESH)
    img=draw_boxes(img,bboxes,scores)

    # Write output
    file_id=get_file_id(args.img_path)
    cv2.imwrite(os.path.join('images', file_id+'_result.png'), img)
