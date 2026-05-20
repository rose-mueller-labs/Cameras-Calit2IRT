'''
w/ Hungarian Matching, velocity prediction, and new flight handling

new changes from backlit_hungarian.py:
  1. Ghost tracking in lost buffer: predicted position advances each frame via EMA velocity,
     so a flying fly's expected landing zone keeps moving with it rather than sitting static.
  4. FLIGHT_DISTANCE_THRESHOLD raised 100 → 350  (119 fps x 15-20 px/frame x 20 lost frames)
  5. Recovery uses ghost position (not stale last-seen bbox) for distance calculation.
  6. ID-swap guard: during Hungarian assignment, a candidate match is rejected when the predicted center lands *closer* to a *different* tracked fly than to the assigned one (prevents neighbor-steal on landing).
  7. MAX_LOST_FRAMES raised 20 to 35 to survive longer flights at 119 fps.
  8. New-fly gate: a detection is only spawned as a new ID when no lost fly's ghost position is within RECOVERY_THRESHOLD, preventing post-landing re-spawns.

https://www.geeksforgeeks.org/dsa/hungarian-algorithm-assignment-problem-set-1-introduction/
https://cp-algorithms.com/graph/hungarian-algorithm.html
--------------------------------------------------
things to fix:
- we need a completely different thing for A types b/c they are colored diffrently and the vid is kinda diff looking
- it's not detecting A's AT ALL rn, so i gotta do that
- same thing with SCO's, make a diff vid for that, because it's detecting 2x the amt that it is for CO's
- when the flies are edging it can't detect them/differentiate them like at all so need to fix that

ideas:
- diff LOWER_ UPPER_ BROWN ranges for SC and AC vids
- diff arena/bg_mask detection bounds for SC and AC vids
- make the bg_mask background a lil looser so that the flies can be seen?
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os

DEBUG = True

# Params
MAX_LOST_FRAMES = 35 # was 20; longer flights at 119 fps need more buffer
MIN_LIFETIME = 5 # frames before an ID is drawn/exported
MAX_PATH_LENGTH = 50

MIN_CONTOUR_AREA = 350 # (same as before — keep as variable per vid if needed)
MAX_CONTOUR_AREA = 2600 # was 1100

# circ ranges
CIRC_MIN = 0.15 # was 0.30
CIRC_MAX = 0.80 # was 0.70

VELOCITY_ALPHA = 0.8 # EMA weight (higher = more reactive)

# distance thresholds calibrated for 119 fps + possible flight
DISTANCE_THRESHOLD = 60 # walking match radius (px)
FLIGHT_DISTANCE_THRESHOLD = 350 # was 100 but now it covers full flight arc across lost frames based on dims

# the value that basicaly controls how many new flies we get detected and how ok we are w swapped ids
# went from 53 to 45 flies by doubling this value in CalitVids/SCO2Ad28.mov
RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2 # same as walking for on-ground recovery

FLYING_SPEED_THRESHOLD = 15 # px/frame speed they gotta go to be flying

STOP_SEC = 10 # seconds of video to process per clip

# vel helpers

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


def get_velocity(obj_id):
    return object_velocities.get(obj_id, (0.0, 0.0))


def predict_position(bbox, obj_id):
    """Return the predicted center for obj_id one frame ahead."""
    cx, cy = get_center(bbox)
    vx, vy = get_velocity(obj_id)
    return (cx + vx, cy + vy)


def predicted_distance(pred_center, curr_bbox):
    """Euclidean distance from a predicted (float) center to a detected bbox center."""
    cx2, cy2 = get_center(curr_bbox)
    return np.sqrt((pred_center[0] - cx2) ** 2 + (pred_center[1] - cy2) ** 2)


def is_flying(obj_id):
    vx, vy = get_velocity(obj_id)
    return np.sqrt(vx ** 2 + vy ** 2) > FLYING_SPEED_THRESHOLD


def size_penalty(bbox1, bbox2):
    """
    Distance-comparable penalty for area mismatch between two bboxes.
    Scaled to be additive with px distances (0 to 50 px equivalent).
    """
    a1 = bbox1[2] * bbox1[3]
    a2 = bbox2[2] * bbox2[3]
    return abs(a1 - a2) / max(a1, a2, 1) * 50


# ghost position fixes

def advance_ghost(obj_id):
    """
    Each frame a fly stays lost, advance its stored ghost position by its last known velocity (with decay), so recovery searches the right area.
    """
    if obj_id not in lost_objects:
        return
    data = lost_objects[obj_id]
    vx, vy = get_velocity(obj_id)
    # decay velocity while lost so ghost doesn't run forever
    object_velocities[obj_id] = (vx * 0.90, vy * 0.90)
    vx, vy = get_velocity(obj_id)

    # update ghost bbox center
    x, y, w, h = data['bbox']
    cx = x + w // 2 + vx
    cy = y + h // 2 + vy
    new_x = int(cx - w // 2)
    new_y = int(cy - h // 2)
    data['bbox'] = (new_x, new_y, w, h)
    data['ghost_center'] = (cx, cy)


def ghost_center(obj_id):
    """Return the ghost (predicted) center for a lost fly."""
    if obj_id in lost_objects:
        return lost_objects[obj_id].get('ghost_center', get_center(lost_objects[obj_id]['bbox']))
    return None


# og helpers

def get_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_unique_color(obj_id):
    if obj_id not in colors:
        rng = random.Random(obj_id * 6364136223846793005 + 1442695040888963407)
        colors[obj_id] = (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))
    return colors[obj_id]


def draw_paths(frame, paths, obj_id):
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        cv2.circle(frame, points[-1], 3, color, -1)


# fg mask

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

    _, fly_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
    fly_mask = cv2.bitwise_and(fly_mask, arena_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fly_mask, arena_mask


# watershed

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
    contours_list = []
    for label in np.unique(markers)[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours_list.append(cnts[0])
    return contours_list


# CONTOURS

def get_good_cnts(contours, frame, arena_mask):
    """
    Filter watershed contours to acc flies.
    """
    large_contours = []
    disp_frm = frame.copy()

    for cnt in contours:
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if cnt_ar < MIN_CONTOUR_AREA or cnt_ar > MAX_CONTOUR_AREA:
            continue

        circularity = (4 * 3.14159 * cnt_ar) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        if arena_mask[cy, cx] == 0:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 165, 255), 2)
            continue

        if circularity > CIRC_MAX or circularity < CIRC_MIN:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 0, 200), 2)
            continue

        cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(disp_frm, f'{round(circularity, 2)}', (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        large_contours.append(cnt)

    return large_contours, disp_frm


# MAIN LOOP

BASE_PATH = "/Volumes/Crucial X9/Downloads/Calit2 Data Collection 05-06-2026"

print(os.listdir(BASE_PATH))

for vid_name in os.listdir(BASE_PATH):
    skip_list = {'.', 'procedure.heic', 'CAO4.MOV'}
    if vid_name[0] == '.' or vid_name in skip_list:
        continue

    vid_path = f"{BASE_PATH}/{vid_name}"
    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]

    csv_name = f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/Tracked_{name}_pwsBacklitV2_fixed_{'debug' if DEBUG else ''}.csv"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = f"./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_pwsBacklitV2_fixed_{'debug' if DEBUG else ''}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # TRACKING DS
    frame_count = 0
    next_object_id = 0
    tracked_objects = {}# id -> bbox
    lost_objects = {} # id -> {bbox, frames_lost, ghost_center}
    object_paths = {} # id -> deque of (cx,cy)
    object_lifetimes = {} # id -> int
    colors = {}

    object_velocities = {} # id -> (vx, vy)  EMA
    last_centers = {} # id -> (cx, cy)

    tracking_data = [] # for csv writing, not other use
    CURRENT_TOTAL_FLIES = 0

    if not cap.isOpened():
        print(f"Error opening {vid_path}")
        continue

    # FRAME
    ret, frame = cap.read()
    if not ret:
        cap.release(); out.release(); continue

    fg_mask, bg_mask = get_fg_mask(frame, name)
    watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame)
    large_contours, _ = get_good_cnts(watershed_cnts, frame, bg_mask)

    for cnt in large_contours:
        bbox = cv2.boundingRect(cnt)
        tracked_objects[next_object_id] = bbox
        object_lifetimes[next_object_id] = 1
        cx, cy = get_center(bbox)
        object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
        object_velocities[next_object_id] = (0.0, 0.0)
        last_centers[next_object_id] = (cx, cy)
        next_object_id += 1

    frame_data = {'frame': 0}
    for obj_id, bbox in tracked_objects.items():
        cx, cy = get_center(bbox)
        frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
    tracking_data.append(frame_data)

    print(f"Starting tracking with {len(tracked_objects)} initial flies detected")
    frame_count = 1

    # MAIN FRAME2
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret or frame_count >= fps * STOP_SEC:
            break
        frame_count += 1

        fg_mask, bg_mask = get_fg_mask(frame2, name)
        # cv2.imwrite(./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_fg_mask_written_pwsBacklitV2_fixed_{'debug' if DEBUG else ''}.png, fg_mask)
        # cv2.imwrite(./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_bg_mask_written_pwsBacklitV2_fixed_{'debug' if DEBUG else ''}.png, bg_mask)
        watershed_cnts = apply_watershed_segmentation(fg_mask, bg_mask, frame2)
        large_contours, disp_frm = get_good_cnts(watershed_cnts, frame2, bg_mask)
        # cv2.imwrite(./2D_Detection/WatershedAlgorithm/Output/Velocity/CalitVids/{name}_disp_frm_written_pwsBacklitV2_fixed_{'debug' if DEBUG else ''}.png, disp_frm)
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

        new_tracked_objects = {}
        used_current = set()
        matched_tracked_ids = set()

        # 0) move ghost positions for all lost flies so that if they moved around swhere else it ok
        for obj_id in list(lost_objects.keys()):
            advance_ghost(obj_id)

        # 1) Hungarian matching
        tracked_ids = list(tracked_objects.keys())

        if tracked_ids and current_bboxes:
            n_tracked = len(tracked_ids)
            n_det = len(current_bboxes)
            cost = np.full((n_tracked, n_det), 1e6, dtype=float)

            for r, obj_id in enumerate(tracked_ids):
                prev_bbox = tracked_objects[obj_id]
                pred_center = predict_position(prev_bbox, obj_id)
                for c, curr_bbox in enumerate(current_bboxes):
                    dist_pred = predicted_distance(pred_center, curr_bbox)
                    dist_direct = calculate_distance(prev_bbox, curr_bbox)
                    if dist_pred < FLIGHT_DISTANCE_THRESHOLD or dist_direct < DISTANCE_THRESHOLD:
                        cost[r, c] = dist_pred + size_penalty(prev_bbox, curr_bbox)

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e6:
                    continue

                obj_id = tracked_ids[r]
                matched_bbox = current_bboxes[c]

# ID-swap fix where we teject the match if another *currently tracked* fly's last position is closer to the detection than obj_id's prediction.
                matched_cx, matched_cy = get_center(matched_bbox)
                pred_cx, pred_cy = predict_position(tracked_objects[obj_id], obj_id)
                dist_to_pred = np.sqrt((matched_cx - pred_cx)**2 + (matched_cy - pred_cy)**2)

                stolen = False
                for other_id, other_bbox in tracked_objects.items():
                    if other_id == obj_id:
                        continue
                    other_cx, other_cy = get_center(other_bbox)
                    dist_other = np.sqrt((matched_cx - other_cx)**2 + (matched_cy - other_cy)**2)
# If another fly's *current* position is closer and obj_id is predicting a large jump, suspect a swap
                    if dist_other < dist_to_pred * 0.5 and dist_to_pred > DISTANCE_THRESHOLD:
                        stolen = True
                        break

                if stolen:
                    # NEW FLY b/c not assigned
                    continue

                new_tracked_objects[obj_id] = matched_bbox
                used_current.add(c)
                matched_tracked_ids.add(obj_id)

                object_lifetimes[obj_id] += 1
                cx, cy = get_center(matched_bbox)
                object_paths[obj_id].append((cx, cy))
                update_velocity(obj_id, (cx, cy))

        # unmatched tracked flies are lost
        for obj_id in tracked_ids:
            if obj_id not in matched_tracked_ids:
                if obj_id not in lost_objects:
                    cx, cy = get_center(tracked_objects[obj_id])
                    lost_objects[obj_id] = {
                        'bbox': tracked_objects[obj_id],
                        'frames_lost': 1,
                        'ghost_center': (float(cx), float(cy)),
                    }
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

        # No tracked objects means that everything is lost
        if not tracked_ids:
            for obj_id in list(tracked_objects.keys()):
                if obj_id not in lost_objects:
                    cx, cy = get_center(tracked_objects[obj_id])
                    lost_objects[obj_id] = {
                        'bbox': tracked_objects[obj_id],
                        'frames_lost': 1,
                        'ghost_center': (float(cx), float(cy)),
                    }
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

        # 2) recovery w the ghost pos now
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

                # use ghost position for recovery search here now instead of just static things
                gc = ghost_center(obj_id)
                if gc is not None:
                    det_cx, det_cy = get_center(curr_bbox)
                    dist = np.sqrt((gc[0] - det_cx)**2 + (gc[1] - det_cy)**2)
                else:
                    dist = calculate_distance(lost_data['bbox'], curr_bbox)

                if dist < best_lost_dist:
                    best_lost_dist = dist
                    best_lost_id = obj_id

            # recovery radius is FLIGHT_DISTANCE_THRESHOLD for previously fast flies, else RECOVERY_THRESHOLD w velocity pred
            if best_lost_id != -1:
                fly_was_fast = (np.sqrt(sum(v**2 for v in get_velocity(best_lost_id))) > FLYING_SPEED_THRESHOLD)
                radius = FLIGHT_DISTANCE_THRESHOLD if fly_was_fast else RECOVERY_THRESHOLD

                if best_lost_dist < radius:
                    new_tracked_objects[best_lost_id] = curr_bbox
                    used_current.add(i)
                    object_lifetimes[best_lost_id] += 1
                    cx, cy = get_center(curr_bbox)
                    object_paths[best_lost_id].append((cx, cy))
                    update_velocity(best_lost_id, (cx, cy))
                    lost_to_remove.append(best_lost_id)
                    print(f"  [RECOVERY] ID {best_lost_id} at dist {round(best_lost_dist, 1)}px after {lost_objects[best_lost_id]['frames_lost']} lost frames{'  [FLIGHT]' if fly_was_fast else ''}")
                    continue

            # truly a new fly if the ghost doesn't work and it wasn't recovery with static lost either
            new_tracked_objects[next_object_id] = curr_bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(curr_bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            object_velocities[next_object_id] = (0.0, 0.0)
            last_centers[next_object_id] = (cx, cy)
            print(f"  [NEW FLY] ID {next_object_id} at ({cx}, {cy})")
            next_object_id += 1

        # 3) Clean up lost buffer to avoid swaps from old things
        for obj_id in lost_to_remove:
            del lost_objects[obj_id]

        expired = [oid for oid, d in lost_objects.items() if d['frames_lost'] > MAX_LOST_FRAMES]
        for obj_id in expired:
            del lost_objects[obj_id]

        tracked_objects = new_tracked_objects

        # data saving
        frame_data = {'frame': frame_count}
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                cx, cy = get_center(bbox)
                frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
        tracking_data.append(frame_data)

        CURRENT_TOTAL_FLIES = len(tracked_objects)

        # drawing
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
                label = f'ID:{obj_id} [FLY]' if is_flying(obj_id) else f'ID:{obj_id}'
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        # draw ghost positions for lost flies (both static lost and with velocity pred)
        if DEBUG:
            for obj_id, data in lost_objects.items():
                gc = ghost_center(obj_id)
                if gc is not None:
                    color = get_unique_color(obj_id)
                    cv2.circle(frame2, (int(gc[0]), int(gc[1])), 8, color, 2)
                    cv2.putText(frame2, f'L:{obj_id}', (int(gc[0]) + 10, int(gc[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        valid_flies = sum(1 for oid in tracked_objects if object_lifetimes.get(oid, 0) >= MIN_LIFETIME)
        cv2.putText(frame2, f'TOTAL FLIES: {valid_flies}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        out.write(frame2)

        if frame_count % 30 == 0:
            print(f"{name} @ frame {frame_count}: valid={valid_flies},tracked={len(tracked_objects)},lost={len(lost_objects)}"
            )

    # write to the CSV
    if tracking_data:
        all_fly_ids = set()
        for fd in tracking_data:
            all_fly_ids.update(k for k in fd if k.startswith('ID_'))
        sorted_fly_ids = sorted(all_fly_ids, key=lambda x: int(x.split('_')[1]))

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['frame'] + sorted_fly_ids
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for fd in tracking_data:
                row = {'frame': fd['frame']}
                for fid in sorted_fly_ids:
                    row[fid] = fd.get(fid, '')
                writer.writerow(row)

        print(f"Total unique fly IDs tracked: {len(sorted_fly_ids)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()