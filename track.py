from networks.detector import HeadDetector
from utils.functions import detect,get_file_id,draw_boxes,is_old,get_box_info,get_box_detail
from config import cfg
# from matplotlib import path
import os
import torch
import cv2
import argparse
import time

THRESH = 0.07
WIDTH, HEITH = 1920,1080
OVIDEO_PATH = '/content/drive/My Drive/DATN/FCHD/videos'
IVIDEO_PATH = '/content/drive/My Drive/DATN/FCHD/videos/entrance_camera_cutted.mp4'

# XLL_LASE = 650
# XLR_LASE = 1080
# YL_LASE = 330
# XHL_LASE = 250
# XHR_LASE = 1450
# YH_LASE = HEITH - 300

OUT_LASE = HEITH - 300
IN_LASE = 330
MAX_DISTANCE= 50

COLORS=[(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(128,128,0),(128,0,128),(0,0,128)]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--video_path", type=str, default=IVIDEO_PATH,
                        help="path of the input video")
    parser.add_argument("-t", "--tracker", type=str, default="kcf",
                        help="OpenCV object tracker type")
    args = parser.parse_args()
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
	  }
    # Load model
    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    model_dict = torch.load(cfg.NEW_MODEL_PATH)['model']
    head_detector.load_state_dict(model_dict)
    head_detector = head_detector.cuda()
    
  # LOad xong fchd 
    # Init parameters
    frame_count = 0
    in_person_number = 0
    out_person_number=0
    obj_cnt = 0
    curr_trackers = []

    # Read video
    vid = cv2.VideoCapture(args.video_path)  

    # Write video
    file_id=get_file_id(args.video_path)
    out = cv2.VideoWriter(os.path.join(OVIDEO_PATH, file_id +'_'+args.tracker+ '_result.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (WIDTH, HEITH))
    t1=time.time()
    while vid.isOpened():
        in_laser_line_color = (0, 0, 255)
        out_laser_line_color = (0, 0, 255)
        tracking_boxes = []

        # Read image from video
        _, frame = vid.read()
        if frame is None:
            break
        # Resize image
        frame = cv2.resize(frame, (WIDTH,HEITH))

        # Check objects in tracker
        old_trackers = curr_trackers
        curr_trackers = []
        for old_tracker in old_trackers:
            # Update tracker
            tracker = old_tracker['tracker']
            (_, box) = tracker.update(frame)
            tracking_boxes.append(box)
            updated_tracker = dict()
            updated_tracker['tracker_id'] = old_tracker['tracker_id']
            updated_tracker['old_boxes'] = old_tracker['old_boxes']
            updated_tracker['tracker'] = tracker

            # Update old boxes
            if len(updated_tracker['old_boxes']) <40:
              updated_tracker['old_boxes'].append(box)
            else:
              updated_tracker['old_boxes'].pop(0)
              updated_tracker['old_boxes'].append(box)

            # Get information of object
            x, y, w, h, center_X, center_Y = get_box_detail(box)

            # Draw rectangle surround object
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[old_tracker['tracker_id']%8], 1)
            cv2.putText(frame, str(old_tracker['tracker_id']), (center_X, center_Y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), thickness=1, lineType=8)
            # Draw old center
            for i in range(0,len(updated_tracker['old_boxes'])):
                x, y, w, h, center_X, center_Y = get_box_detail(updated_tracker['old_boxes'][i])
                cv2.circle(frame,(center_X,center_Y),int(i/2),COLORS[old_tracker['tracker_id']%8],-1)
            # Compare center of object and laser line
            if center_Y > OUT_LASE:
                out_laser_line_color = (0, 255, 255)
                out_person_number += 1
                print('detect person get out %d th in frame %d ' %(out_person_number,frame_count))
            elif center_Y < IN_LASE:
                in_laser_line_color = (0, 255, 255)
                in_person_number += 1
                print('detect person get in %d th in frame %d ' %(in_person_number,frame_count))
            else:
                # Otherwise continue track object
                curr_trackers.append(updated_tracker)

        # Detect object every 5 frame
        if frame_count % 5 == 0:
            # Detect object
            boxes_d,scores = detect(frame,head_detector,THRESH)
            # print('detect boxs ', boxes_d)
            # Check box in boxes_d
            for box_d,score in zip(boxes_d,scores):
                xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box_d)
                
                # print(get_box_info(box_d[:4]))
                # If center of object add max distance smaller than laser line
                # p = path.Path([(XHL_LASE, YH_LASE),(XLL_LASE,YL_LASE), (XLR_LASE,YL_LASE),(XHR_LASE, YH_LASE)])
                # if p.contains_point((center_Xd,center_Yd)):
                if center_Yd > IN_LASE and center_Yd < OUT_LASE:
                    # If object is new
                    if not is_old(center_Xd,center_Yd, tracking_boxes, MAX_DISTANCE):
                        # Draw rectangle surround object
                        cv2.rectangle(frame, (xd, yd), (xd + wd, yd + hd), (0, 255, 0), 1)
                        # Draw score at top of object
                        cv2.putText(frame, '{:.3f}'.format(score), (xd, yd - 9), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (0, 0, 255), thickness=1, lineType=8)
                        # Creat new tracker
                        tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                        obj_cnt += 1
                        new_tracker = dict()
                        bbox = [xd, yd, wd, hd]
                        old_boxes=[]
                        tracker.init(frame, tuple(bbox))
                        new_tracker['tracker_id'] = obj_cnt
                        new_tracker['tracker'] = tracker
                        new_tracker['old_boxes']= old_boxes
                        curr_trackers.append(new_tracker)

        # Increase frame
        frame_count += 1

        # Display number of person get in, get out
        cv2.putText(frame, "Get in person: " + str(in_person_number), (10, IN_LASE - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
        #cv2.putText(frame, "Press Esc to quit", (10, laser_line + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Get out person: " + str(out_person_number), (10, OUT_LASE + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
        # Draw laser line
        # cv2.line(frame, (XLL_LASE, YL_LASE), (XLR_LASE, YL_LASE), out_laser_line_color, 2)
        # cv2.line(frame, (XHL_LASE, YH_LASE), (XHR_LASE, YH_LASE), in_laser_line_color, 2)
        # cv2.line(frame, (XLL_LASE, YL_LASE), (XHL_LASE, YH_LASE), (0, 0, 255), 2)
        # cv2.line(frame, (XLR_LASE, YL_LASE), (XHR_LASE, YH_LASE), (0, 0, 255), 2)
        cv2.line(frame, (0, IN_LASE), (WIDTH, IN_LASE), in_laser_line_color, 2)
        cv2.line(frame, (0, OUT_LASE), (WIDTH, OUT_LASE), out_laser_line_color, 2)
        # Display and write frame
        
        # cv2.putText(frame, "FPS: " + str(1/(t2-t1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 0), 2)
        out.write(frame)
        # cv2.imshow("Image", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27:
        #     break
    t2=time.time()
    print("running time: ",t2-t1)
    print(frame_count)
    vid.release()
    out.release()
