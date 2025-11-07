import cv2
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import csv
import os

name = "vial_closeup"
vid_path=f"SampleVideos/{name}.mp4"
cap = cv2.VideoCapture(vid_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = f'{name}_written.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# |timestamp | timestamp ...
# -----------------------...
# |bb coor   | bb coor   ...
# -----------------------...
# with open(f"{name}_written_coordinates.csv", "w", newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Timestamp{i}' * len()])

subtractor_name=""
backSub = cv2.createBackgroundSubtractorKNN() # cv2.createBackgroundSubtractorMOG2()
prev_frame = None
if not cap.isOpened():
    print("Error opening video file")
    exit()
else:
    ret, frame1 = cap.read()
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame2 = cap.read()
        if ret:
            # Apply background subtraction
            prev_frame = frame2
            fg_mask = backSub.apply(frame1)
            fg_mask = backSub.apply(frame2)
        if not ret:
            break

        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_ct = cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        frame_ct = cv2.drawContours(frame2, contours, -1, (0, 255, 0), 2)

        retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        min_contour_area = 10
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 200), 3)
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 200), 3)
            # compare to frame1, see which fly is closest
            # "saved_image.jpg", image
            cv2.imwrite('frame1.jpg', frame1)
            cv2.imwrite('frame2.jpg', frame2)
        
        out.write(frame1)
        out.write(frame2)
        frame1 = frame2

cap.release()
out.release()
cv2.destroyAllWindows()