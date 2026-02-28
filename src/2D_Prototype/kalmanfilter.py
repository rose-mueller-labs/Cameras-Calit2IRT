import cv2
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import csv
import os
import random

name = "4k 60fps.MXF"
vid_path = f"SampleVideos/{name}"
cap = cv2.VideoCapture(vid_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = f'DistanceComparisonOutputs/{name}_written_kf.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

backSub = cv2.createBackgroundSubtractorKNN()

next_object_id = 0
tracks = {} # id -> {'kf': kalman, 'bbox': (x,y,w,h), 'missed': int}
colors = {} # id -> color

def get_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w/2.0, y + h/2.0], dtype=np.float32)

def get_unique_color(obj_id):
    if obj_id not in colors:
        while True:
            c = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            if c not in colors.values():
                colors[obj_id] = c
                break
    return colors[obj_id]

def create_kalman(initial_center):
    # State: [x, y, vx, vy]
    kf = cv2.KalmanFilter(4, 2)
    dt = 120.0 # assumes 1 frame per time unit; you can use 1/fps

    # F
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=np.float32)

    # H
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Q, R, P
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    kf.statePost = np.array([[initial_center[0]],
                             [initial_center[1]],
                             [0.0],
                             [0.0]], dtype=np.float32)
    return kf

if not cap.isOpened():
    print("Error")
    exit()

ret, frame1 = cap.read()
if not ret:
    print("Empty video")
    exit()

# initialize from first frame detections
fg_mask = backSub.apply(frame1)
contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 30
max_contour_area = 500
large_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

for cnt in large_contours:
    bbox = cv2.boundingRect(cnt)
    center = get_center(bbox)
    kf = create_kalman(center)
    tracks[next_object_id] = {'kf': kf, 'bbox': bbox, 'missed': 0}
    next_object_id += 1

max_dist = 50.0 # distance threshold for association
max_missed = 10 # frames to keep an unobserved track

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    fg_mask = backSub.apply(frame2)

    retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 10
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

    # prediction step for all tracks
    predicted_centers = {}
    for obj_id, tr in tracks.items():
        pred = tr['kf'].predict()  # 4x1
        predicted_centers[obj_id] = np.array([pred[0,0], pred[1,0]], dtype=np.float32)

    # associate detections with predictions (greedy)
    unmatched_detections = set(range(len(current_bboxes)))
    new_tracks = {}

    # for each track, find nearest detection
    for obj_id, tr in tracks.items():
        if len(unmatched_detections) == 0:
            tr['missed'] += 1
            if tr['missed'] <= max_missed:
                new_tracks[obj_id] = tr
            continue

        pred_center = predicted_centers[obj_id]
        best_det = -1
        best_dist = 1e9

        for i in unmatched_detections:
            det_center = get_center(current_bboxes[i])
            dist = np.linalg.norm(det_center - pred_center)
            if dist < best_dist:
                best_dist = dist
                best_det = i

        if best_det != -1 and best_dist < max_dist:
            # update with measurement
            det_center = get_center(current_bboxes[best_det])
            measurement = det_center.reshape(2, 1).astype(np.float32)
            tr['kf'].correct(measurement)
            tr['bbox'] = current_bboxes[best_det]
            tr['missed'] = 0
            new_tracks[obj_id] = tr
            unmatched_detections.remove(best_det)
        else:
            # no good match, just keep predicted state, increment missed
            tr['missed'] += 1
            if tr['missed'] <= max_missed:
                new_tracks[obj_id] = tr

    # start new tracks for remaining detections
    for i in unmatched_detections:
        bbox = current_bboxes[i]
        center = get_center(bbox)
        kf = create_kalman(center)
        new_tracks[next_object_id] = {'kf': kf, 'bbox': bbox, 'missed': 0}
        next_object_id += 1

    tracks = new_tracks

    # draw results
    for obj_id, tr in tracks.items():
        x, y, w, h = tr['bbox']
        color = get_unique_color(obj_id)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame2, f'ID:{obj_id}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # todo: write (frame_idx, obj_id, x_center, y_center, etc.) to CSV here

    out.write(frame2)

cap.release()
out.release()
cv2.destroyAllWindows()
