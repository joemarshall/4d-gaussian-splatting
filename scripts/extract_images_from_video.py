import argparse
import os
from pathlib import Path

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--root_dir", default="", help="input path to the video")

    args = parser.parse_args()
    return args
    
args = parse_args()

root_dir = args.root_dir
videos = [os.path.join(root_dir, vname) for vname in os.listdir(args.root_dir) if vname.endswith(".mp4")]
for video in videos:
    cam_name = video.split('/')[-1].split('.')[-2]
    images = os.path.join(root_dir, "images/") # override args.images

    if not os.path.exists(images):
        os.makedirs(images)
        
    cam_video = cv2.VideoCapture(video)
    # frame
    currentframe = 0
    
    while ( True ):
        # reading from frame
        ret, frame = cam_video.read()
    
        if ret:
            # 如果视频仍然存在，继续创建图像
            name = os.path.join(images, cam_name + '_{:04d}.png'.format(currentframe))
            print ('Creating...' + name)
    
            # 写入提取的图像
            cv2.imwrite(name, frame)

            # 增加计数器，以便显示创建了多少帧
            currentframe += 1
        else :
            break
        
    cam_video.release()
    cv2.destroyAllWindows()