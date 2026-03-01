import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def debug(frame, title="Debug"):
    if len(frame) == 4:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(frame[0])
        ax[0, 1].imshow(frame[1])
        ax[1, 0].imshow(frame[2])
        ax[1, 1].imshow(frame[3])

        ax[0, 0].set_title('Vial 1')
        ax[0, 1].set_title('Vial 2')
        ax[1, 0].set_title('Vial 3')
        ax[1, 1].set_title('Vial 4')

        plt.show()

    else:
        rgimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(rgimg)
        plt.show()

def average_rgb(frame):
    mean_b, mean_g, mean_r = cv2.mean(frame)[:3]
    average_color = (mean_r, mean_g, mean_b)
    return average_color

name = 'Recording from 2026-02-11 13-11-28.511852.webm'
vid_path = f"SampleVideos/{name}"
output_path = f'FlyFalling/Outputs/{name}_written.mp4'
csv_name   = f'FlyFalling/Tracked_{name}.csv'

NUM_VIALS = 4
MIN_FLY_AREA = 8
MAX_FLY_AREA = 300
DARK_OFFSET = 40 # how many intensity units below local mean counts as "dark"
DEBUG = True 

os.makedirs('FlyFalling/Outputs', exist_ok=True)

cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
frame_index = 0

csv_rows = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_s = frame_index / fps
    counts = 0

    frame = frame[100:, 100:1350]

    if DEBUG:
        debug(frame, "frame")

    
    vial1 = frame[:, 100:450]
    vial2 = frame[:, 400:700]
    vial3 = frame[:, 680:900]
    vial4 = frame[:, 930:]

    vial_frames = [vial1, vial2, vial3, vial4]

    th3_frames = []

    debug(vial_frames)
    for vial in vial_frames:
        avg_vial_rgb = average_rgb(vial)
        print(avg_vial_rgb)
        debug(vial)
        # detect fly (dark blobs) and the background is whitish grayish here

    
    debug(th3_frames)
        
cap.release()
out.release()
cv2.destroyAllWindows()
