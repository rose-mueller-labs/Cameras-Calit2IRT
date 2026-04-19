'''
Tutorial used:
https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/

Issues found:
- Flies being redected after inactivity as new flies
- Flies sometimes being changed into new ID flies for seemingly no reason, why is that happening
- Zig-zaggy paths, this is due to the matching algorithm -> make it use velocity to see the changes

Solutions implemented:
- Added path tracking and visualization
- Filter out objects that appear for less than 5 frames
- Improved ID consistency with temporal tracking and lost object recovery
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
import matplotlib.pyplot as plt

BASE_PATH="/Volumes/Crucial X9/Cameras-Calit2IRT/src/SampleVideos/Backlit"

save_flies = False
LOWER_BROWN = np.array([0,  85,  0])
UPPER_BROWN = np.array([215, 200, 138])

for vid_path, min_contour_area in [
        # (f"{BASE_PATH}/2k 120fps backlit.MXF", 30),
        # (f"{BASE_PATH}/4k 24fps.MXF", 30),
        # (f"{BASE_PATH}/4k 60fps.MXF", 30), # src/SampleVideos/4k 60fps.MXF
        # (f"{BASE_PATH}/120fps 2K.MXF", 30),
        (f"{BASE_PATH}/4k 30fps box.MOV", 30)
        # (f"{BASE_PATH}/180fps 2K.MXF", 30),
        # (f"{BASE_PATH}/180fps more flys.MXF", 30)
        ]:
    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]
    csv_name = f"./2D_Detection/WatershedAlgorithm/Output/Backlit/Tracked_{name}_pwsBacklit.csv"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = f'./2D_Detection/WatershedAlgorithm/Output/Backlit/{name}_pwsBacklit.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0

    if not cap.isOpened():
        print("Error opening video file")
        exit()
    else:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret or frame_count <= fps * 5:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # The interior is very bright white
                _, white_region = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Find the largest contour = the white arena interior
                contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                arena_mask = np.zeros_like(gray)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
                    # Erode slightly to avoid picking up edge artifacts
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                    arena_mask = cv2.erode(arena_mask, kernel, iterations=1)
                # cv2.imwrite("arena_mask.png", arena_mask)
                # plt.show()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fly_mask = cv2.inRange(rgb, LOWER_BROWN, UPPER_BROWN)
                
                # Restrict to arena interior only
                fly_mask = cv2.bitwise_and(fly_mask, arena_mask)
                
                # Clean up noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                fg_mask = fly_mask
                cv2.imwrite("debug_mask.png", fly_mask)
            if not ret or frame_count >= fps * 5:
                break

            contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"{len(contours)} contours found @ frame {frame_count}")
            frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            min_contour_area = 10
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            # cv2.imwrite("detect.png", frame)
            frame_count += 1
            out.write(frame)
    print("Finished detecting. Total flies found : ")

    cap.release()
    out.release() 
    cv2.destroyAllWindows()