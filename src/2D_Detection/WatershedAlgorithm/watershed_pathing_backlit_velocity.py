'''
**TODO**:
- In petri dish backlit videos since there's no tape to distinguish the end, the borders on the right
    get detected as contours/part of it. --> need to tweak the arena_mask to remove large contours
    or decrease max_contour_size after we get contours from the fly_mask.

**CHANGES**:
- Added `object_velocities` dict: stores a rolling weighted average velocity (vx, vy) per fly ID.
- `predict_position(bbox, obj_id)`: predicts where a fly will be next frame using its velocity.
- Matching in Step 1 now compares detections against the PREDICTED position, not the raw last bbox. 
    This means a fly that suddenly accelerates during takeoff still matches its old ID as long as the predicted 
    position is within FLIGHT_DISTANCE_THRESHOLD of the detection.
- Two-tier threshold:
    DISTANCE_THRESHOLD (90px) for  normal/walking matching (tight)
    FLIGHT_DISTANCE_THRESHOLD (250px) like a relaxed matching used against the predicted position, for fast-moving / flying flies.
- Velocity is updated every frame using exponential smoothing (alpha=0.4 favors recent motion).
- Lost-object recovery still uses RECOVERY_THRESHOLD as before, but now also tries the predicted position of the lost 
    fly when searching for a match.
- `is_flying(obj_id)` is a heuristic a fly is considered airborne when its speed exceeds
    FLYING_SPEED_THRESHOLD px/frame. Flying flies are labelled "[FLY]" on the output video so we can see when it decides 
    this.
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
import matplotlib.pyplot as plt
import math
import os

# Velocity helpers

def update_velocity(obj_id, new_center):
    """Exponential moving average of (vx, vy) per fly."""
    if obj_id in object_velocities:
        prev_cx, prev_cy = last_centers[obj_id]
        new_vx = new_center[0] - prev_cx
        new_vy = new_center[1] - prev_cy
        old_vx, old_vy = object_velocities[obj_id]
        object_velocities[obj_id] = (
            VELOCITY_ALPHA * new_vx + (1 - VELOCITY_ALPHA) * old_vx,
            VELOCITY_ALPHA * new_vy + (1 - VELOCITY_ALPHA) * old_vy,
        )
    else:
        object_velocities[obj_id] = (0.0, 0.0)
    last_centers[obj_id] = new_center


def predict_position(bbox, obj_id):
    """Return the predicted center for obj_id one frame ahead."""
    cx, cy = get_center(bbox)
    if obj_id in object_velocities:
        vx, vy = object_velocities[obj_id]
        return (cx + vx, cy + vy)
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

# older helpers

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
        crop_path = (
            f"./2D_Detection/WatershedAlgorithm/Output/Velocity/"
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
    _, white_region = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (53, 53))
        arena_mask = cv2.erode(arena_mask, kernel, iterations=1)
    cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/arena_mask_{name}.png", arena_mask)

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
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_paths(frame, paths, obj_id):
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        cv2.circle(frame, points[-1], 3, color, -1)


def get_good_cnts(contours, frame):
    large_contours = []
    disp_frm = frame.copy()
    for cnt in contours:
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * 3.14 * cnt_ar) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        thing = frame[y:y + h, x:x + w]
        avg_color_per_row = np.average(thing, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        if cnt_ar < min_contour_area or cnt_ar > MAX_CONTOUR_AREA:
            continue
        if circularity > 0.70 or circularity < 0.30:
            continue

        disp_frm = cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 0, 200), 3)
        disp_frm = cv2.putText(disp_frm, f'{avg_color}', (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        large_contours.append(cnt)

    return large_contours, disp_frm

# Constants

BASE_PATH = "/Volumes/Crucial X9/Downloads/Calit2 Data Collection 05-06-2026"

LOWER_BROWN = np.array([0, 70, 0])
UPPER_BROWN = np.array([215, 185, 185])

MAX_LOST_FRAMES = 10 # frames to keep a lost object in the recovery buffer
MIN_LIFETIME = 5 # min frames before an object is considered valid
MAX_PATH_LENGTH = 50 # max points in path history

MAX_CONTOUR_AREA = 1100
STOP_SEC = 5

# Velocity / flight parameters added
VELOCITY_ALPHA = 0.4  # EMA weight for new velocity samples (higher = more reactive)
FLYING_SPEED_THRESHOLD = 15  # px/frame; above this a fly is labelled as flying
#  Normal walking match uses DISTANCE_THRESHOLD (tight).
#  When velocity predicts a large displacement we allow FLIGHT_DISTANCE_THRESHOLD (loose).
#  FLIGHT_DISTANCE_THRESHOLD should be large enough to cover max realistic flight speed per frame.
FLIGHT_DISTANCE_THRESHOLD = 75 # px: adjust based on the fps & arena size in videos (100 -> 250 -> good)

CURRENT_TOTAL_FLIES = 0

print(os.listdir(BASE_PATH))
# vid_names = ['ACO1.MOV', 'ACO2.MOV', 'ACO3.MOV', 'ACO4.MOV', 'ACO5.MOV', 
#               #'CACO4.MOV', 'CACO5.MOV', 
#               'CO1.MOV', 'CO2.MOV', 'CO3.MOV',
#               'CO4.MOV', 'CO5.MOV']
# vid_names = ['ACO2.MOV', 'ACO3.MOV', 'CO2.MOV', 'CO3.MOV']
for vid_name in os.listdir(BASE_PATH)[:2]:
    skip_list = {'.', 'procedure.heic', 'CAO4.MOV', }
    if vid_name[0] == '.' or vid_name in skip_list:
        continue

    vid_path = f"{BASE_PATH}/{vid_name}"

    DISTANCE_THRESHOLD = 200
    LOW_COL = 140
    HIGH_COL = 1600
    LOWER_ROW = 800
    HIGH_ROW = 2900
    min_contour_area = 200

    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]
    csv_name = (f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/"
                f"Tracked_{name}_pwsBacklitV.csv")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = (f'./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/'
                   f'{name}_pwsBacklitV.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2
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

    # NEW: velocity state
    object_velocities = {}  # obj_id -> (vx, vy) EMA
    last_centers = {} # obj_id -> (cx, cy) from the previous frame

    tracking_data = []

    if not cap.isOpened():
        print("Error opening video file")
        continue

    ret, frame = cap.read()
    
    # frame = frame[LOWER_ROW:HIGH_ROW, LOW_COL:HIGH_COL]
    if frame_count >= fps * STOP_SEC:
        cap.release()
        out.release()
        continue
    if ret or frame_count <= fps * STOP_SEC:
        fg_mask, bg_mask = get_fg_mask(frame, name)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_bg_mask_pwsBacklitV.png", bg_mask)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_mask_pwsBacklitV.png", fg_mask)
    if not ret:
        cap.release()
        out.release()
        continue

    watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame)
    contours = watershed_cnts
    retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    large_contours, disp_frm = get_good_cnts(contours, frame)

    frame_ct = frame.copy()
    for cnt in large_contours:
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        tracked_objects[next_object_id] = bbox
        object_lifetimes[next_object_id] = 1
        cx, cy = get_center(bbox)
        object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
        # Initialise velocity to zero
        object_velocities[next_object_id] = (0.0, 0.0)
        last_centers[next_object_id] = (cx, cy)
        frame_ct = cv2.rectangle(frame_ct, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Label flying flies so you can spot mis-matches visually
        label = f'fli'
        cv2.putText(frame_ct, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # write it = frame_ct
    
    frame_ct_rgb = cv2.cvtColor(frame_ct, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_ct_rgb)
    plt.savefig(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_cnt.png")

    frame_data = {'frame': 0}
    for obj_id, bbox in tracked_objects.items():
        cx, cy = get_center(bbox)
        frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
    tracking_data.append(frame_data)

    print(f"Starting tracking with {len(tracked_objects)} initial flies detected")
    frame_count += 1

    # every frame after first
    while cap.isOpened():
        ret, frame2 = cap.read()
        if frame_count >= fps * STOP_SEC:
            break
        if not ret or frame_count >= fps * STOP_SEC:
            break
        # frame2 = frame2[LOWER_ROW:HIGH_ROW, LOW_COL:HIGH_COL]
        frame_count += 1

        fg_mask, bg_mask = get_fg_mask(frame2, name)
        cv2.imwrite(f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_debug_mask_pwsBacklitV.png", fg_mask)

        watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame2)
        contours = watershed_cnts
        retval, mask_thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        large_contours, disp_frm = get_good_cnts(contours, frame2)
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

        new_tracked_objects = {}
        used_current = set()

        # Step 1: Match current detections to actively tracked objects.
        # For each tracked fly we compute its PREDICTED position one frame
        # ahead using its velocity.  We then accept a match if the detection
        # is within FLIGHT_DISTANCE_THRESHOLD of that prediction.  We also
        # keep a tighter "direct" distance check (DISTANCE_THRESHOLD) so that
        # slow / stationary flies are still matched conservatively.

        for obj_id, prev_bbox in tracked_objects.items():
            if len(current_bboxes) == 0:
                break

            pred_center = predict_position(prev_bbox, obj_id)

            best_match_i = -1
            best_score = float('inf')   # lower = better

            for i, curr_bbox in enumerate(current_bboxes):
                if i in used_current:
                    continue

                # Distance from predicted position to detection
                dist_pred = predicted_distance(pred_center, curr_bbox)
                # Direct (last-known-position) distance as a fallback
                dist_direct = calculate_distance(prev_bbox, curr_bbox)

                # Accept if either:
                # a) walking-speed match: direct dist < DISTANCE_THRESHOLD
                # b) flight-speed match: predicted dist < FLIGHT_DISTANCE_THRESHOLD
                if dist_pred < FLIGHT_DISTANCE_THRESHOLD or dist_direct < DISTANCE_THRESHOLD:
                    # Score = predicted distance (prefers whoever is closest to prediction)
                    score = dist_pred
                    if score < best_score:
                        best_score = score
                        best_match_i = i

            if best_match_i != -1:
                matched_bbox = current_bboxes[best_match_i]
                new_tracked_objects[obj_id] = matched_bbox
                used_current.add(best_match_i)
                object_lifetimes[obj_id] += 1
                cx, cy = get_center(matched_bbox)
                object_paths[obj_id].append((cx, cy))
                # Update velocity with the actual new center
                update_velocity(obj_id, (cx, cy))
            else:
                # Fly not matched = save last known position for recovery
                if obj_id not in lost_objects:
                    lost_objects[obj_id] = {'bbox': prev_bbox, 'frames_lost': 1}
                else:
                    lost_objects[obj_id]['frames_lost'] += 1
                # Keep velocity decaying slightly while lost so the prediction
                # doesn't grow unbounded
                if obj_id in object_velocities:
                    vx, vy = object_velocities[obj_id]
                    object_velocities[obj_id] = (vx * 0.85, vy * 0.85)

        # Step 2: For each unmatched detection, check lost_objects first. 
        # Also use predicted position of lost fly for recovery.
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

                # Try both: last-known position and velocity-predicted position
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
                print(f"  [RECOVERY] fly ID {best_lost_id} recovered at distance "
                      f"{best_lost_dist:.1f}px after "
                      f"{lost_objects[best_lost_id]['frames_lost']} lost frames")
            else:
                # Truly new fly
                new_tracked_objects[next_object_id] = curr_bbox
                object_lifetimes[next_object_id] = 1
                cx, cy = get_center(curr_bbox)
                object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
                object_velocities[next_object_id] = (0.0, 0.0)
                last_centers[next_object_id] = (cx, cy)
                next_object_id += 1

        # Step 3: Clean up lost buffer
        for obj_id in lost_to_remove:
            del lost_objects[obj_id]

        expired = [obj_id for obj_id, data in lost_objects.items()
                   if data['frames_lost'] > MAX_LOST_FRAMES]
        for obj_id in expired:
            del lost_objects[obj_id]

        tracked_objects = new_tracked_objects

        # Store frame data
        frame_data = {'frame': frame_count}
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                cx, cy = get_center(bbox)
                frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
        tracking_data.append(frame_data)

        CURRENT_TOTAL_FLIES = len(tracked_objects)

        if save_flies:
            save_fly_crops(frame2, tracked_objects, object_lifetimes, frame_count, name)

        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                frame2 = cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
                # Label flying flies so you can spot mis-matches visually
                label = f'ID:{obj_id} [FLY]' if is_flying(obj_id) else f'ID:{obj_id}'
                if is_flying(obj_id):
                    print(f'  [FLY] fly ID {obj_id}')
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        cv2.putText(frame2, f'TOTAL FLIES: {CURRENT_TOTAL_FLIES}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        out.write(frame2)

        if frame_count % 30 == 0:
            valid_flies = sum(1 for obj_id in tracked_objects
                              if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
            print(f"{name} @ {frame_count} frames with {valid_flies} valid flies "
                  f"(total tracked: {len(tracked_objects)}, "
                  f"in lost buffer: {len(lost_objects)})")

    # Write CSV
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