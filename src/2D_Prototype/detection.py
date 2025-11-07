import cv2
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import csv
import os

name = "vial_closeup"
vid_path = f"SampleVideos/{name}.mp4"
cap = cv2.VideoCapture(vid_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = f'{name}_written.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

subtractor_name = ""
backSub = cv2.createBackgroundSubtractorKNN()
prev_frame = None

# Object tracking variables
next_object_id = 0
tracked_objects = {}  # Maps object_id to (x, y, w, h)
colors = {}  # Maps object_id to color

def get_center(bbox):
    x, y, w, h = bbox
    return (x + w//2, y + h//2)

def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_unique_color(obj_id):
    if obj_id not in colors:
        blug = [
                (255, 0, 0),
                (0, 255, 0)
            ]
        colors[obj_id] = blug[obj_id % 5]
    return colors[obj_id]

if not cap.isOpened():
    print("Error")
    exit()
else:
    ret, frame1 = cap.read()
    
    # Initialize first frame objects
    fg_mask = backSub.apply(frame1)
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 10
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Assign IDs to objects in first frame
    for cnt in large_contours:
        bbox = cv2.boundingRect(cnt)
        tracked_objects[next_object_id] = bbox
        next_object_id += 1
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame2 = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        prev_frame = frame2
        fg_mask = backSub.apply(frame2)
        
        # Threshold and morphological operations
        retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 10
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # Get bounding boxes for current frame
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]
        
        # Match current objects to tracked objects
        new_tracked_objects = {}
        used_current = set()
        
        # For each tracked object, find closest match in current frame
        for obj_id, prev_bbox in tracked_objects.items():
            if len(current_bboxes) == 0:
                break
                
            min_dist = float('inf')
            best_match_idx = -1
            
            for idx, curr_bbox in enumerate(current_bboxes):
                if idx in used_current:
                    continue
                dist = calculate_distance(prev_bbox, curr_bbox)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx
            
            # Assign match if found and distance is reasonable
            if best_match_idx != -1 and min_dist < 100:  # Distance threshold
                new_tracked_objects[obj_id] = current_bboxes[best_match_idx]
                used_current.add(best_match_idx)
        
        # Assign new IDs to unmatched flies
        for idx, curr_bbox in enumerate(current_bboxes):
            if idx not in used_current:
                new_tracked_objects[next_object_id] = curr_bbox
                next_object_id += 1
        
        tracked_objects = new_tracked_objects
        
        # Draw bounding boxes with unique colors for each fly ID
        for obj_id, bbox in tracked_objects.items():
            x, y, w, h = bbox
            color = get_unique_color(obj_id)
            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 3)
            # Draw ID label
            cv2.putText(frame2, f'ID:{obj_id}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite('frame1.jpg', frame1)
        cv2.imwrite('frame2.jpg', frame2)
        
        out.write(frame2)
        frame1 = frame2.copy()

cap.release()
out.release()
cv2.destroyAllWindows()