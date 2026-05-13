import cv2
import numpy as np
import random
import csv
from collections import deque
import math
import matplotlib.pyplot as plt

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

def get_name(obj_id):
    names = ['Alden', 'Neela', 'Shreya', 'Zach', 'Corrina', 'Jacob', 'Miao', 'Alex', 'Irene',
             'Michael', 'Laurence', 'Vivian']
    return names[obj_id % len(names)]

def get_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def apply_watershed_segmentation(sure_fg, sure_bg, original_frame):
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
    _, white_region = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (53, 53))
        arena_mask = cv2.erode(arena_mask, kernel, iterations=1)

    cv2.imwrite(f"./Output/Backlit/arena_mask_{name}.png", arena_mask)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fly_mask = cv2.inRange(rgb, LOWER_BROWN, UPPER_BROWN)
    fly_mask = cv2.bitwise_and(fly_mask, arena_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fly_mask, arena_mask

def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_paths(frame, paths, obj_id, y_offset=0, x_offset=0):
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        shifted = [(x + x_offset, y + y_offset) for x, y in points]
        for i in range(len(shifted) - 1):
            cv2.line(frame, shifted[i], shifted[i + 1], color, 2)
        cv2.circle(frame, shifted[-1], 3, color, -1)

def get_good_cnts(contours, frame):
    large_contours = []
    disp_frm = frame.copy()
    for i, cnt in enumerate(contours):
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * 3.14 * cnt_ar) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        thing = frame[y:y+h, x:x+w]
        if thing.size == 0:
            continue
        avg_color_per_row = np.average(thing, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        if cnt_ar < min_contour_area or cnt_ar > MAX_CONTOUR_AREA:
            continue
        if circularity > 0.70 or circularity < 0.30:
            continue

        disp_frm = cv2.rectangle(disp_frm, (x, y), (x+w, y+h), (0, 0, 200), 3)
        disp_frm = cv2.putText(disp_frm, f'{avg_color}', (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        large_contours.append(cnt)

    return large_contours, disp_frm

LOWER_BROWN = np.array([0, 0, 0])
UPPER_BROWN = np.array([90, 90, 90])
MAX_LOST_FRAMES = 10
MIN_LIFETIME = 5
MAX_PATH_LENGTH = 50
DISTANCE_THRESHOLD = 200
RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2
min_contour_area = 20
MAX_CONTOUR_AREA = 80
Y_CROP = 121
X_CROP_END = 1356

CURRENT_TOTAL_FLIES = 0

cap = cv2.VideoCapture(0)
name = 'live'
csv_name = f"./Output/Backlit/UROPVids/Tracked_{name}_pwsBacklit.csv"

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# -Y_CROP
fps = 120
output_path = f'./Output/Backlit/UROPVids/{name}_pwsBacklit.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
next_object_id = 0
tracked_objects = {}
lost_objects = {}
object_paths = {}
object_lifetimes = {}
object_was_stationary = {}
colors = {}
tracking_data = []

if not cap.isOpened():
    print("Error opening video file")
    exit()

while cap.isOpened():
    ret, frame_full = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # plt.imshow(cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB))
    # plt.show()

    frame_crop = frame_full[Y_CROP:, :X_CROP_END].copy()

    fg_mask, bg_mask = get_fg_mask(frame_crop)
    cv2.imwrite(f"./Output/Backlit/UROPVids/{name}_debug_mask_pwsBacklit.png", fg_mask)

    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours, disp_frm = get_good_cnts(contours, frame_crop)
    current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

    new_tracked_objects = {}
    used_current = set()

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
            if obj_id not in lost_objects:
                lost_objects[obj_id] = {'bbox': prev_bbox, 'frames_lost': 1}
            else:
                lost_objects[obj_id]['frames_lost'] += 1

    lost_to_remove = []
    for i, curr_bbox in enumerate(current_bboxes):
        if i in used_current:
            continue

        best_lost_id = -1
        best_lost_dist = float('inf')

        for obj_id, lost_data in lost_objects.items():
            if lost_data['frames_lost'] > MAX_LOST_FRAMES:
                continue
            if obj_id in new_tracked_objects:
                continue
            dist = calculate_distance(lost_data['bbox'], curr_bbox)
            if dist < best_lost_dist:
                best_lost_dist = dist
                best_lost_id = obj_id

        if best_lost_id != -1 and best_lost_dist < RECOVERY_THRESHOLD:
            new_tracked_objects[best_lost_id] = curr_bbox
            used_current.add(i)
            object_lifetimes[best_lost_id] += 1
            cx, cy = get_center(curr_bbox)
            object_paths[best_lost_id].append((cx, cy))
            lost_to_remove.append(best_lost_id)
            print(f"  [RECOVERY] fly ID {best_lost_id} recovered at distance {best_lost_dist:.1f}px after {lost_objects[best_lost_id]['frames_lost']} lost frames")
        else:
            new_tracked_objects[next_object_id] = curr_bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(curr_bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            next_object_id += 1

    for obj_id in lost_to_remove:
        del lost_objects[obj_id]

    expired = [obj_id for obj_id, data in lost_objects.items() if data['frames_lost'] > MAX_LOST_FRAMES]
    for obj_id in expired:
        del lost_objects[obj_id]

    tracked_objects = new_tracked_objects

    frame_data = {'frame': frame_count}
    for obj_id, bbox in tracked_objects.items():
        if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
            cx, cy = get_center(bbox)
            frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
    tracking_data.append(frame_data)

    CURRENT_TOTAL_FLIES = len(tracked_objects)

    for obj_id, bbox in tracked_objects.items():
        if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
            x, y, w, h = bbox
            x_full = x
            y_full = y + Y_CROP

            draw_paths(frame_full, object_paths, obj_id, y_offset=Y_CROP, x_offset=0)
            color = get_unique_color(obj_id)
            cv2.rectangle(frame_full, (x_full, y_full), (x_full + w, y_full + h), color, 3)
            cv2.putText(frame_full, f'{get_name(obj_id)}', (x_full, max(30, y_full - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    cv2.putText(frame_full, f'TOTAL FLIES: {CURRENT_TOTAL_FLIES}', (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    if frame_count % 30 == 0:
        valid_flies = sum(1 for obj_id in tracked_objects if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
        print(f"{name} @ {frame_count} frames with {valid_flies} valid flies (total tracked: {len(tracked_objects)}, in lost buffer: {len(lost_objects)})")

    cv2.imshow("LiveFeed", frame_full)
    out.write(frame_full)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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