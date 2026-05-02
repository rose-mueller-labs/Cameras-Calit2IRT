'''
TODO:
- In petri dish backlist videos since there's no tape to distinguish the end, the borders on the right
get detected as contours/part of it. --> need to tweak the arena_mask to remove large contours
or decrease max_contour_size after we get contours from the fly_mask.
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
import matplotlib.pyplot as plt
import math
import os


def save_fly_crops(frame, tracked_objects, object_lifetimes, frame_count, name):
        """SAVE THE FLIES."""
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) < MIN_LIFETIME:
                continue
            x, y, w, h = bbox
            # Clamp to frame boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_path = (
                f"./2D_Detection/WatershedAlgorithm/Output/Backlit/"
                f"fly_crop_{name}_frame{frame_count}_ID{obj_id}.png"
            )
            cv2.imwrite(crop_path, crop)

def get_unique_color(obj_id):
    if obj_id not in colors:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        if (r, g, b) not in colors.values():
            colors[obj_id] = (r, g, b)
        else:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors[obj_id] = (r, g, b)
    return colors[obj_id]

def get_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def apply_watershed_segmentation(sure_fg, sure_bg, original_frame):
        """
        https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/
        1. Noise removal with morphological operations
        2. Sure background detection (dilation)
        3. Distance transform for sure foreground
        4. Unknown region calculation
        5. Marker labeling
        6. Watershed application
        """
            
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        if len(original_frame.shape) == 2:
            original_frame_3ch = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
        else:
            original_frame_3ch = original_frame.copy()

        markers = cv2.watershed(original_frame_3ch, markers)
        labels = np.unique(markers)
        contours_list = []

        for label in labels[2:]:
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contours_list.append(contours[0])

        return contours_list

def get_fg_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The interior is very bright white
    _, white_region = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)
    
    # Find the largest contour = the white arena interior
    contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
        # Erode slightly to avoid picking up edge artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (53, 53))
        arena_mask = cv2.erode(arena_mask, kernel, iterations=1)
    cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Backlit/arena_mask_{name}.png", arena_mask)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fly_mask = cv2.inRange(rgb, LOWER_BROWN, UPPER_BROWN)
    
    # Restrict to arena interior only
    fly_mask = cv2.bitwise_and(fly_mask, arena_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fly_mask, arena_mask

def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_paths(frame, paths, obj_id):
    """Draw the path history for a given object"""
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        for i in range(len(points) - 1):
            thickness = 2
            cv2.line(frame, points[i], points[i + 1], color, thickness)
        cv2.circle(frame, points[-1], 3, color, -1)

def get_good_cnts(contours, frame):
    large_contours = []
    disp_frm = frame.copy()
    for i, cnt in enumerate(contours):
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * 3.14 * cnt_ar) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        eccentricity = (math.sqrt(abs(w**2-h**2)))/w

        thing = frame[y:y+h,x:x+w]
        avg_color_per_row = np.average(thing, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        brightness = np.mean(avg_color)

        if cnt_ar < min_contour_area or cnt_ar > MAX_CONTOUR_AREA:
            continue
        if perimeter == 0:
            continue
        if circularity > 0.70 or circularity < 0.30:
            continue
    
        disp_frm = cv2.rectangle(disp_frm, (x, y), (x+w, y+h), (0, 0, 200), 3)
        disp_frm = cv2.putText(disp_frm, f'{avg_color}', (x+10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)    
    
        large_contours.append(cnt)

    return large_contours, disp_frm

# BASE_PATH="/Volumes/Crucial X9/Cameras-Calit2IRT/src/SampleVideos/Backlit"
BASE_PATH="/Volumes/Crucial X9/Downloads/UROP Data Colletion 4-26-2026"

LOWER_BROWN = np.array([0,  70,   0])
UPPER_BROWN = np.array([215, 175, 138]) # works for all flies except the one in the rightside
MAX_LOST_FRAMES = 10 # how many frames to keep a moving lost object
MIN_LIFETIME = 5 # min frames an object must be seen to be considered valid
MAX_PATH_LENGTH = 50 # max number of points to keep in path history
# DISTANCE_THRESHOLD = 90 # 90 to minimize flying = new ID issue.

MAX_CONTOUR_AREA = 1100

STOP_SEC = 60

CURRENT_TOTAL_FLIES = 0

# for vid_path, min_contour_area, LOWER_BOUND_CROP, DISTANCE_THRESHOLD in [
#         # (f"{BASE_PATH}/2k 120fps backlit.MXF", 20, -1, 200),
#         # (f"{BASE_PATH}/4k 30fps box.MOV", 20, -1, 200), # got perfect results
        
#         # (f"{BASE_PATH}/4k 30fps Petri dish.MOV", 20, -1, 200),
#         # (f"{BASE_PATH}/4k 120fps Petri dish.MOV", 20, -1, 200),
        
#         # (f"{BASE_PATH}/4k 120fps box.MOV", 350, 3000, 90) # TODO: Velocity implementation for flying
#         ]:

print(os.listdir(BASE_PATH))
for vid_name in os.listdir(BASE_PATH):
    if vid_name[0] == '.' or vid_name == 'ACO1.MOV' or vid_name == 'procedure.heic' or vid_name == 'CACO4_short.MOV' or vid_name == 'CO3.MOV' or vid_name == 'ACO5.MOV' or vid_name == 'CAO4.MOV':
        continue
    # vid_name = 'CO2.MOV'
    vid_path = f"{BASE_PATH}/{vid_name}"
    
    DISTANCE_THRESHOLD = 90 # CAO5
    LOWER_BOUND_CROP = 2747
    UPPER_BOUND_CROP = 1200
    WIDTH_BOUND = 1600
    WIDTH_L_BOUND = 300
    min_contour_area = 200
    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]
    csv_name = f"./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/Tracked_{name}_pwsBacklit.csv"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = f'./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/{name}_pwsBacklit.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH_BOUND-WIDTH_L_BOUND, LOWER_BOUND_CROP-UPPER_BOUND_CROP))
    
    # if WIDTH_BOUND != -1:
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH_BOUND, frame_height))
    
    # if LOWER_BOUND_CROP != -1:
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, LOWER_BOUND_CROP))

    save_flies = False

    mid = total_frames // 2
    snapshot_frames = {mid, mid + 1}
    
    frame_count = 0

    # Object tracking dictionaries
    next_object_id = 0
    tracked_objects = {} # key = object_id, value = (x, y, w, h)
    lost_objects = {} # Temporarily store lost objects for recovery
    object_paths = {} # key = object_id, value = deque of (cx, cy) positions
    object_lifetimes = {} # key = object_id, value = number of frames seen
    object_was_stationary = {}  # key = object_id, value = bool (was stationary when lost)
    colors = {} # key = object_id, value = color

    # To store into CSV the different coordinates across frames and fly IDs
    tracking_data = [] # List of dictionaries, one per frame

    if not cap.isOpened():
        print("Error opening video file")
        exit()
    else:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if LOWER_BOUND_CROP != -1:
        #     frame = frame[:LOWER_BOUND_CROP, :]
        # if WIDTH_BOUND != -1:
        #     frame = frame[:, :WIDTH_BOUND]
        # rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(rgbframe)
        # plt.show()
        frame = frame[UPPER_BOUND_CROP:LOWER_BOUND_CROP, WIDTH_L_BOUND:WIDTH_BOUND]
        # rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(rgbframe)
        # plt.show()
        if ret or frame_count <= fps * STOP_SEC:
            fg_mask, bg_mask = get_fg_mask(frame)
            cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/{name}_bg_mask_pwsBacklit.png", bg_mask)
            cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/{name}_debug_mask_pwsBacklit.png", fg_mask)
        if not ret or frame_count >= fps * STOP_SEC:
            break
            
        watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame)
        
        # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = watershed_cnts
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        large_contours, disp_frm = get_good_cnts(contours, frame)
        # plt.imshow(cv2.cvtColor(disp_frm, cv2.COLOR_BGR2RGB))
        # plt.show()

        for cnt in large_contours:
            bbox = cv2.boundingRect(cnt)
            tracked_objects[next_object_id] = bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            next_object_id += 1
            x, y, w, h = bbox
            frame_written = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/{name}_cnt_mask_pwsBacklit.png", frame_written)

        frame_data = {'frame': 0}
        for obj_id, bbox in tracked_objects.items():
            cx, cy = get_center(bbox)
            frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
        tracking_data.append(frame_data)

        print(f"Starting tracking with {len(tracked_objects)} initial flies detected")
        
        frame_count += 1

        while cap.isOpened():
            ret, frame2 = cap.read()
            if not ret or frame_count >= fps * STOP_SEC:
                break
            # if LOWER_BOUND_CROP != -1:
            #     frame2 = frame2[:LOWER_BOUND_CROP, :]
            # if WIDTH_BOUND != -1:
            #     frame2 = frame2[:, :WIDTH_BOUND]
            frame2 = frame2[UPPER_BOUND_CROP:LOWER_BOUND_CROP, WIDTH_L_BOUND:WIDTH_BOUND]
            frame_count += 1

            fg_mask, bg_mask = get_fg_mask(frame2)
            cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Backlit/UROPVids/{name}_debug_mask_pwsBacklit.png", fg_mask)

            watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame2)
            
            # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = watershed_cnts
            frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            large_contours, disp_frm = get_good_cnts(contours, frame2)
            
            # plt.imshow(cv2.cvtColor(disp_frm, cv2.COLOR_BGR2RGB))
            # plt.show()
            current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

            new_tracked_objects = {} # from now on, what's different?

            used_current = set()

            # Step 1: Match current detections to actively tracked objects
            for obj_id, prev_bbox in tracked_objects.items():
                if len(current_bboxes) == 0:
                    break

                min_dist = float('inf')
                best_match_i = -1

                for i, curr_bbox in enumerate(current_bboxes):
                    if i in used_current:
                        continue
                    dist = calculate_distance(prev_bbox, curr_bbox)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_i = i

                if best_match_i != -1 and min_dist < DISTANCE_THRESHOLD:
                    new_tracked_objects[obj_id] = current_bboxes[best_match_i]
                    used_current.add(best_match_i)
                    object_lifetimes[obj_id] += 1
                    cx, cy = get_center(current_bboxes[best_match_i])
                    object_paths[obj_id].append((cx, cy))
                else:
                    # Fly not matched = save its last known position for potential recovery
                    if obj_id not in lost_objects:
                        lost_objects[obj_id] = {'bbox': prev_bbox, 'frames_lost': 1}
                    else:
                        lost_objects[obj_id]['frames_lost'] += 1

            # Step 2: For each unmatched detection, check lost_objects first before assigning a brand new ID.
            lost_to_remove = []
            for i, curr_bbox in enumerate(current_bboxes):
                if i in used_current:
                    continue

                best_lost_id = -1
                best_lost_dist = float('inf')

                for obj_id, lost_data in lost_objects.items():
                    # Skip flies already expired or already recovered this frame
                    if lost_data['frames_lost'] > MAX_LOST_FRAMES:
                        continue
                    if obj_id in new_tracked_objects:
                        continue
                    dist = calculate_distance(lost_data['bbox'], curr_bbox)
                    if dist < best_lost_dist:
                        best_lost_dist = dist
                        best_lost_id = obj_id

                if best_lost_id != -1 and best_lost_dist < RECOVERY_THRESHOLD:
                    # Close enough = recover the old ID
                    new_tracked_objects[best_lost_id] = curr_bbox
                    used_current.add(i)
                    object_lifetimes[best_lost_id] += 1
                    cx, cy = get_center(curr_bbox)
                    object_paths[best_lost_id].append((cx, cy))
                    lost_to_remove.append(best_lost_id)
                    print(f"  [RECOVERY] fly ID {best_lost_id} recovered at distance "
                          f"{best_lost_dist:.1f}px after "
                          f"{lost_objects[best_lost_id]['frames_lost']} lost frames")
                else:
                    # Truly new fly = assign a fresh ID
                    new_tracked_objects[next_object_id] = curr_bbox
                    object_lifetimes[next_object_id] = 1
                    cx, cy = get_center(curr_bbox)
                    object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
                    next_object_id += 1

            # Step 3: Remove recovered flies from lost buffer; expire stale ones
            for obj_id in lost_to_remove:
                del lost_objects[obj_id]

            expired = [obj_id for obj_id, data in lost_objects.items() if data['frames_lost'] > MAX_LOST_FRAMES]
            for obj_id in expired:
                del lost_objects[obj_id]

            tracked_objects = new_tracked_objects

            # Store current frame data (only for objects with sufficient lifetime)
            frame_data = {'frame': frame_count}
            for obj_id, bbox in tracked_objects.items():
                if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                    cx, cy = get_center(bbox)
                    frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
            tracking_data.append(frame_data)

            # Draw paths and bounding boxes (only for objects with sufficient lifetime)
            CURRENT_TOTAL_FLIES = len(tracked_objects)

            if save_flies:
                save_fly_crops(frame2, tracked_objects, object_lifetimes, frame_count, name)

            for obj_id, bbox in tracked_objects.items():
                if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                    draw_paths(frame2, object_paths, obj_id)
                    x, y, w, h = bbox
                    color = get_unique_color(obj_id)
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 3)

                    cv2.putText(frame2, f'ID:{obj_id}', (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            cv2.putText(frame2, f'TOTAL FLIES: {CURRENT_TOTAL_FLIES}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

            out.write(frame2)

            if frame_count % 30 == 0:
                valid_flies = sum(1 for obj_id in tracked_objects
                                  if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
                print(f"{name} @ {frame_count} frames with {valid_flies} valid flies "
                      f"(total tracked: {len(tracked_objects)}, "
                      f"in lost buffer: {len(lost_objects)})")
            # out.write(frame)
    # Write tracking data to CSV after processing all frames
    if tracking_data:
        all_fly_ids = set()
        for frame_data in tracking_data:
            all_fly_ids.update([k for k in frame_data.keys() if k.startswith('ID_')])

        sorted_fly_ids = sorted(all_fly_ids, key=lambda x: int(x.split('_')[1]))

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['frame'] + sorted_fly_ids
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for frame_data in tracking_data:
                row = {'frame': frame_data['frame']}
                for fly_id in sorted_fly_ids:
                    row[fly_id] = frame_data.get(fly_id, '')
                writer.writerow(row)

        print(f"Total unique flies tracked: {len(sorted_fly_ids)}")

    cap.release()
    out.release() 
    cv2.destroyAllWindows()