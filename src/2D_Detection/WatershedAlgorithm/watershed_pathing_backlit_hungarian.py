'''
w/ Hungarian Matching & velocity
https://www.geeksforgeeks.org/dsa/hungarian-algorithm-assignment-problem-set-1-introduction/
https://cp-algorithms.com/graph/hungarian-algorithm.html
https://en.wikipedia.org/wiki/Hungarian_algorithm
https://www.columbia.edu/~cs2035/courses/ieor8100.F12/lec6.pdf
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import math
import os

DEBUG = True

def update_velocity(obj_id, new_center):
    """Exponential moving average of (vx, vy) per fly."""
    if obj_id in object_velocities:
        prev_cx, prev_cy = last_centers[obj_id]
        new_vx = new_center[0] - prev_cx
        new_vy = new_center[1] - prev_cy
        old_vx, old_vy = object_velocities[obj_id]
        object_velocities[obj_id] = (VELOCITY_ALPHA * new_vx + (1 - VELOCITY_ALPHA) * old_vx, VELOCITY_ALPHA * new_vy + (1 - VELOCITY_ALPHA) * old_vy,)
    else:
        object_velocities[obj_id] = (0.0, 0.0)
    last_centers[obj_id] = new_center


def predict_position(bbox, obj_id):
    """Return the predicted center for obj_id one frame ahead."""
    cx, cy = get_center(bbox)
    if obj_id in object_velocities:
        vx, vy = object_velocities[obj_id]
        return (cx+vx, cy+vy)
    return (cx, cy)


def predicted_distance(pred_center, curr_bbox):
    """Euclidean distance from a predicted (float) center to a detected bbox center."""
    cx2, cy2 = get_center(curr_bbox)
    return np.sqrt((pred_center[0] - cx2) ** 2 + (pred_center[1] - cy2) ** 2)


def is_flying(obj_id):
    """True when the fly's recent speed exceeds the flying threshold."""
    if obj_id not in object_velocities:
        return False
    vx, vy = object_velocities[obj_id]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    return speed > FLYING_SPEED_THRESHOLD


def size_penalty(bbox1, bbox2):
    """
    Returns a distance-comparable penalty based on how different two bboxes are in area.
    0 when identical, up to around 50px when areas are completely mismatched.
    Helps avoid swapping IDs when two flies cross at similar distances.
    """
    a1 = bbox1[2] * bbox1[3]
    a2 = bbox2[2] * bbox2[3]
    return abs(a1 - a2) / max(a1, a2, 1) * 50  # scaled to be px-comparable

# Older helpers

def save_fly_crops(frame, tracked_objects, object_lifetimes, frame_count, name):
    """SAVE THE FLIES."""
    for obj_id, bbox in tracked_objects.items():
        if object_lifetimes.get(obj_id, 0) < MIN_LIFETIME:
            continue
        x, y, w, h = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_path = f"./2D_Detection/WatershedAlgorithm/Output/Velocity/fly_crop_{name}_frame{frame_count}_ID{obj_id}_backlitV2.png"
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


def get_fg_mask(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_region = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (53, 53))
        arena_mask = cv2.erode(arena_mask, kernel, iterations=1)

    cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/arena_mask_{name}_V2.png", arena_mask)

    # detect dark blobs on white background — no color range needed
    _, fly_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV) # dark pixels become white
    fly_mask = cv2.bitwise_and(fly_mask, arena_mask) # restrict to arena interior

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fly_mask, arena_mask


def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_paths(frame, paths, obj_id):
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        cv2.circle(frame, points[-1], 3, color, -1)


def get_good_cnts(contours, frame, arena_mask):  # <-- add arena_mask param
    large_contours = []
    disp_frm = frame.copy()
    for cnt in contours:
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if cnt_ar < min_contour_area or cnt_ar > MAX_CONTOUR_AREA:
            continue
        circularity = (4 * 3.14 * cnt_ar) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # Reject if center falls outside the eroded arena (boundary/tape region)
        if arena_mask[cy, cx] == 0:
            disp_frm = cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 165, 255), 3) # orange = boundary reject
            continue

        if circularity > 0.70 or circularity < 0.30:
            disp_frm = cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 0, 200), 3) # red = circularity reject
            continue

        disp_frm = cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 200, 0), 3) # green = accepted
        disp_frm = cv2.putText(disp_frm, f'{circularity:.2f}', (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        large_contours.append(cnt)

    return large_contours, disp_frm

# Constants

BASE_PATH = "/Volumes/Crucial X9/Downloads/Calit2 Data Collection 05-06-2026"

LOWER_BROWN = np.array([0, 70, 0])
UPPER_BROWN = np.array([200, 185, 185])

MAX_LOST_FRAMES = 20  # raised from 10: more forgiving for brief occlusions
MIN_LIFETIME = 5 # min frames before an object is considered valid
MAX_PATH_LENGTH = 50 # max points in path history

MAX_CONTOUR_AREA = 1100
STOP_SEC = 10

# Velocity / flight parameters
VELOCITY_ALPHA = 0.8 # EMA weight for new velocity samples (higher = more reactive)
FLYING_SPEED_THRESHOLD = 15 # px/frame; above this a fly is labelled as flying
FLIGHT_DISTANCE_THRESHOLD = 100 # px: relaxed threshold for predicted-position matching
DISTANCE_THRESHOLD = 50 # walking threshold

CURRENT_TOTAL_FLIES = 0

# Main loop
print(os.listdir(BASE_PATH))

for vid_name in os.listdir(BASE_PATH):
    skip_list = {'.', 'procedure.heic', 'CAO4.MOV'}
    if vid_name[0] == '.' or vid_name in skip_list:
        continue

    vid_path = f"{BASE_PATH}/{vid_name}"

    min_contour_area = 350

    # Make them the same to avoid spawning new ids
    RECOVERY_THRESHOLD = DISTANCE_THRESHOLD

    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]
    csv_name = (f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/Tracked_{name}_pwsBacklitV2_{'debug' if DEBUG else ''}.csv")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(f"FPS: {fps}") 119 FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = (f'./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_pwsBacklitV2_{'debug' if DEBUG else ''}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    save_flies = False

    mid = total_frames // 2
    snapshot_frames = {mid, mid + 1}

    frame_count = 0

    # Tracking state
    next_object_id = 0
    tracked_objects = {}
    lost_objects = {}
    object_paths = {}
    object_lifetimes = {}
    colors = {}

    # Velocity state
    object_velocities = {} # obj_id -> (vx, vy) EMA
    last_centers = {} # obj_id -> (cx, cy) from the previous frame

    tracking_data = []

    if not cap.isOpened():
        print("Error opening video file")
        continue

    ret, frame = cap.read()

    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.show()

    if frame_count >= fps * STOP_SEC:
        cap.release()
        out.release()
        continue
    if ret or frame_count <= fps * STOP_SEC:
        fg_mask, bg_mask = get_fg_mask(frame, name)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_bg_mask_pwsBacklitV2_{'debug' if DEBUG else ''}.png", bg_mask)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_mask_pwsBacklitV2_{'debug' if DEBUG else ''}.png", fg_mask)
    if not ret:
        cap.release()
        out.release()
        continue

    watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame)
    contours = watershed_cnts
    retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    large_contours, disp_frm = get_good_cnts(contours, frame, bg_mask)

    # cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_disp_frm_pwsBacklitV2_{'debug' if DEBUG else ''}.png", disp_frm)

    frame_ct = frame.copy()
    for cnt in large_contours:
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        tracked_objects[next_object_id] = bbox
        object_lifetimes[next_object_id] = 1
        cx, cy = get_center(bbox)
        object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
        object_velocities[next_object_id] = (0.0, 0.0)
        last_centers[next_object_id] = (cx, cy)
        next_object_id += 1
        frame_ct = cv2.rectangle(frame_ct, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame_ct, 'fli', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    frame_ct_rgb = cv2.cvtColor(frame_ct, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_ct_rgb)
    plt.savefig(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_cnt_pwsBacklitV2_{'debug' if DEBUG else ''}.png")

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

        frame_count += 1

        fg_mask, bg_mask = get_fg_mask(frame2, name)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_mask_pwsBacklitV2_{'debug' if DEBUG else ''}.png", fg_mask)

        watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame2)
        contours = watershed_cnts
        retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        large_contours, disp_frm = get_good_cnts(contours, frame2, bg_mask)
        # cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_disp_frm_pwsBacklitV2_{'debug' if DEBUG else ''}.png", disp_frm)
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

        new_tracked_objects = {}
        used_current = set()

        # Step 1: Hungarian matching
        tracked_ids = list(tracked_objects.keys())

        if tracked_ids and current_bboxes:
            n_tracked = len(tracked_ids)
            n_det = len(current_bboxes)
            # create cost matrix 
            cost = np.full((n_tracked, n_det), 1e6, dtype=float) # row = IDs, col = detections
            # init w/ big cost so that we don't get them assigned

            for r, obj_id in enumerate(tracked_ids):
                prev_bbox = tracked_objects[obj_id]
                pred_center = predict_position(prev_bbox, obj_id)
                for c, curr_bbox in enumerate(current_bboxes):
                    dist_pred = predicted_distance(pred_center, curr_bbox)
                    dist_direct = calculate_distance(prev_bbox, curr_bbox)
                    # Only consider feasible pairs
                    if dist_pred < FLIGHT_DISTANCE_THRESHOLD or dist_direct < DISTANCE_THRESHOLD:
                        cost[r, c] = dist_pred + size_penalty(prev_bbox, curr_bbox) # predicted_distance + size_penalty

            row_ind, col_ind = linear_sum_assignment(cost) # finds the globally optimal assignment

            matched_tracked_ids = set()
            for r, c in zip(row_ind, col_ind):
                # if cost is impossible/non-sensical we move on
                if cost[r, c] >= 1e6:
                    continue 
                # resolve the actual object ID and the bounding box it's been matched to in curr frame
                obj_id = tracked_ids[r]
                matched_bbox = current_bboxes[c]
                # update the tracker's state with the new bbox & mark both sides as "used" so leftover unmatched items can be handled separately
                new_tracked_objects[obj_id] = matched_bbox
                used_current.add(c)
                matched_tracked_ids.add(obj_id)
                # old stuff from greedy alg
                object_lifetimes[obj_id] += 1
                cx, cy = get_center(matched_bbox)
                object_paths[obj_id].append((cx, cy))
                update_velocity(obj_id, (cx, cy))

            # any tracked fly with no match goes to the lost buffer
            for obj_id in tracked_ids:
                if obj_id not in matched_tracked_ids:
                    if obj_id not in lost_objects:
                        lost_objects[obj_id] = {'bbox': tracked_objects[obj_id], 'frames_lost': 1}
                    else:
                        lost_objects[obj_id]['frames_lost'] += 1
                    # decay the velocity so ids don't grow unbounded
                    if obj_id in object_velocities:
                        vx, vy = object_velocities[obj_id]
                        object_velocities[obj_id] = (vx * 0.85, vy * 0.85)
        else:
            # no tracked objects or no detections so move everything to lost
            for obj_id in tracked_ids:
                if obj_id not in lost_objects:
                    lost_objects[obj_id] = {'bbox': tracked_objects[obj_id], 'frames_lost': 1}
                else:
                    lost_objects[obj_id]['frames_lost'] += 1
                if obj_id in object_velocities:
                    vx, vy = object_velocities[obj_id]
                    object_velocities[obj_id] = (vx * 0.85, vy * 0.85)

        # Step 2: Recovery
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

                dist_direct = calculate_distance(lost_data['bbox'], curr_bbox)
                pred_center = predict_position(lost_data['bbox'], obj_id)
                dist_pred = predicted_distance(pred_center, curr_bbox)
                dist = min(dist_direct, dist_pred)

                if dist < best_lost_dist:
                    best_lost_dist = dist
                    best_lost_id = obj_id

            if best_lost_id != -1 and best_lost_dist < RECOVERY_THRESHOLD:
                new_tracked_objects[best_lost_id] = curr_bbox
                used_current.add(i)
                object_lifetimes[best_lost_id] += 1
                cx, cy = get_center(curr_bbox)
                object_paths[best_lost_id].append((cx, cy))
                update_velocity(best_lost_id, (cx, cy))
                lost_to_remove.append(best_lost_id)
                print(f"  [RECOVERY] fly ID {best_lost_id} recovered at distance {best_lost_dist:.1f}px after {lost_objects[best_lost_id]['frames_lost']} lost frames")
            else:
                # Truly new fly
                new_tracked_objects[next_object_id] = curr_bbox
                object_lifetimes[next_object_id] = 1
                cx, cy = get_center(curr_bbox)
                object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
                object_velocities[next_object_id] = (0.0, 0.0)
                last_centers[next_object_id] = (cx, cy)
                print(f"  [NEW FLY] fly ID {next_object_id} at ({cx}, {cy})")
                next_object_id += 1

        # Step 3: Clean up lost buffer
        for obj_id in lost_to_remove:
            del lost_objects[obj_id]

        expired = [obj_id for obj_id, data in lost_objects.items()
                   if data['frames_lost'] > MAX_LOST_FRAMES]
        for obj_id in expired:
            del lost_objects[obj_id]

        tracked_objects = new_tracked_objects

        # saving to write later
        frame_data = {'frame': frame_count}
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                cx, cy = get_center(bbox)
                frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
        tracking_data.append(frame_data)

        CURRENT_TOTAL_FLIES = len(tracked_objects)

        if save_flies:
            save_fly_crops(frame2, tracked_objects, object_lifetimes, frame_count, name)

        # drawing
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                frame2 = cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
                label = f'ID:{obj_id} [FLY]' if is_flying(obj_id) else f'ID:{obj_id}'
                if is_flying(obj_id):
                    print(f'  [FLY] fly ID {obj_id}')
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        cv2.putText(frame2, f'TOTAL FLIES: {CURRENT_TOTAL_FLIES}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        out.write(frame2)

        if frame_count % 30 == 0:
            valid_flies = sum(1 for obj_id in tracked_objects if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
            print(f"{name} @ {frame_count} frames with {valid_flies} valid flies (total tracked: {len(tracked_objects)}, in lost buffer: {len(lost_objects)})")

    # writing
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