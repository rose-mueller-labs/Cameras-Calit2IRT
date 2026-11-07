'''
Fly tracker (blue-border videos), detection from the HIGH-FREQUENCY content of each frame
--------------------------------------------------
detection, per frame:
  1. low-pass  : Gaussian blur of the gray frame (keeps only the smooth/low-frequency part)
  2. high-freq : gray - lowpass. Large values = edges / small detail = where flies' outlines are (and the blue border / wall edges)
  3. threshold the high-freq map (dark side of each edge only by default -> the fly body, not the halo around it)
  4. cv2.inRange on HSV -> blue mask, dilated, subtracted out so the border's own edge doesnt count
  5. close the fly's outline into one solid blob, contours -> filter by area + aspect (long thin edge lines are rejected) -> fly boxes
tracking is unchanged: Hungarian matching, EMA velocity, ghost positions, ID-swap guard, new-fly gate.
--------------------------------------------------
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
SHOW_PREVIEW = True # plt window of the 1st frame per video; close it to continue

# detection params
LOWPASS_SIGMA = 8 # px: Gaussian sigma of the low-pass. Roughly the fly's half-size: anything sharper/smaller than this counts as 'high frequency'
HF_THRESH = 30 # gray levels of (lowpass - gray) to count as a high-frequency fly edge. The first-frame report prints the 99.9th percentile to help tune
HF_DARK_ONLY = True # True: only keep where the pixel is DARKER than its low-pass (the fly body). False: use |high-pass| (both sides of every edge, so boxes come out bigger)
CLOSE_KERNEL = 9 # px: closes the fly's outline / merges its pieces into one blob
BLUE_LO = (100, 40, 40) # HSV (OpenCV: H 0-179). Blue is ~120
BLUE_HI = (145, 255, 255)
BLUE_DILATE = 15 # px: grown blue mask is removed from the high-freq map so the border's own edge isnt a 'fly'
MIN_CONTOUR_AREA = 80
MAX_CONTOUR_AREA = 1500
MAX_ASPECT = 5.0 # rotated-box length/width. Edge lines (border, wall strip) are long and thin, flies are ovals
CIRC_MIN = 0.45 # circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle). A 25x10 oval is ~0.65-0.75; ragged / stringy border pieces are much lower. Check the printed values on the 1st frame
CIRC_MAX = 0.8 # tiny pixelated contours can read slightly over 1.0
HF_VIS_MAX = 100 # only for the preview: residual that shows as full white

# tracking params
MAX_LOST_FRAMES = 35
MIN_LIFETIME = 5
MAX_PATH_LENGTH = 50
VELOCITY_ALPHA = 0.8
DISTANCE_THRESHOLD = 60
FLIGHT_DISTANCE_THRESHOLD = 350
RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2
FLYING_SPEED_THRESHOLD = 15
STOP_SEC = 150


# detection

def get_masks(frame, verbose=False):
    """Returns (fly_mask, blue_mask, lowpass_u8, highfreq_vis_u8)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), LOWPASS_SIGMA) # low-pass filter
    high = gray - low # what the low-pass removed = high frequencies
    source = np.clip(-high, 0, None) if HF_DARK_ONLY else np.abs(high)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array(BLUE_LO), np.array(BLUE_HI))
    blue = cv2.dilate(blue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLUE_DILATE, BLUE_DILATE)))

    hf_mask = (source > HF_THRESH).astype(np.uint8) * 255
    hf_mask = cv2.bitwise_and(hf_mask, cv2.bitwise_not(blue))
    fly_mask = cv2.morphologyEx(hf_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL)))

    if verbose:
        outside = source[blue == 0]
        print(f"  high-freq residual outside the blue: 99% = {np.percentile(outside, 99):.1f}, 99.9% = {np.percentile(outside, 99.9):.1f}, max = {outside.max():.1f}  (HF_THRESH = {HF_THRESH})")

    hf_vis = np.clip(np.abs(high) * 255.0 / HF_VIS_MAX, 0, 255).astype(np.uint8)
    return fly_mask, blue, np.clip(low, 0, 255).astype(np.uint8), hf_vis


def segment_frame(frame, verbose=False):
    fly_mask, blue, low, hf_vis = get_masks(frame, verbose)
    contours, _ = cv2.findContours(fly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # external contour = filled blob, outline interior included
    disp_frm = frame.copy()
    good = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (255, 120, 0), 2)
            if verbose:
                print(f"  [REJECT area] ({cx},{cy}) area={round(area)} (allowed {MIN_CONTOUR_AREA}-{MAX_CONTOUR_AREA})")
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        if aspect > MAX_ASPECT:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (200, 0, 200), 2)
            if verbose:
                print(f"  [REJECT aspect] ({cx},{cy}) area={round(area)} aspect={round(aspect, 1)} (allowed <= {MAX_ASPECT})")
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        if circularity < CIRC_MIN or circularity > CIRC_MAX:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 0, 200), 2)
            if verbose:
                print(f"  [REJECT circ] ({cx},{cy}) area={round(area)} circ={round(circularity, 2)} (allowed {CIRC_MIN}-{CIRC_MAX})")
            continue
        if verbose:
            print(f"  [KEEP] ({cx},{cy}) area={round(area)} aspect={round(aspect, 1)} circ={round(circularity, 2)}")
        cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 200, 0), 2)
        good.append(cnt)
    return good, disp_frm, fly_mask, blue, low, hf_vis


def show_preview(frame, low, hf_vis, fly_mask, disp_frm, name):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, im, title in zip(
        axes,
        [frame, low, hf_vis, fly_mask, disp_frm],
        ['frame', f'low-pass (sigma={LOWPASS_SIGMA})', '|high-freq| = |gray - lowpass|', 'thresholded, blue removed, closed',
         'contours (green=kept, orange=area, purple=aspect, red=circ)'],
    ):
        if im.ndim == 2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


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
    cx, cy = get_center(bbox)
    vx, vy = get_velocity(obj_id)
    return (cx + vx, cy + vy)


def predicted_distance(pred_center, curr_bbox):
    cx2, cy2 = get_center(curr_bbox)
    return np.sqrt((pred_center[0] - cx2) ** 2 + (pred_center[1] - cy2) ** 2)


def is_flying(obj_id):
    vx, vy = get_velocity(obj_id)
    return np.sqrt(vx ** 2 + vy ** 2) > FLYING_SPEED_THRESHOLD


def size_penalty(bbox1, bbox2):
    """Distance-comparable penalty for area mismatch between two bboxes (0 to 50 px equivalent)."""
    a1 = bbox1[2] * bbox1[3]
    a2 = bbox2[2] * bbox2[3]
    return abs(a1 - a2) / max(a1, a2, 1) * 50


# ghost position fixes

def advance_ghost(obj_id):
    if obj_id not in lost_objects:
        return
    data = lost_objects[obj_id]
    vx, vy = get_velocity(obj_id)
    object_velocities[obj_id] = (vx * 0.90, vy * 0.90)
    vx, vy = get_velocity(obj_id)

    x, y, w, h = data['bbox']
    cx = x + w // 2 + vx
    cy = y + h // 2 + vy
    data['bbox'] = (int(cx - w // 2), int(cy - h // 2), w, h)
    data['ghost_center'] = (cx, cy)


def ghost_center(obj_id):
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


# MAIN LOOP

BASE_PATH = "/Volumes/Crucial X9/Cameras-Calit2IRT/MateChoice_Detection/TestingSet/CroppedSet"
RESULTS_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/MateChoice_Detection/TestingOutput"
DEBUG_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/MateChoice_Detection/TestingDebug"

# print(os.listdir(BASE_PATH))

for vid_name in os.listdir(BASE_PATH):

    skip_list = {}

    if vid_name[0] == '.' or vid_name in skip_list:
        continue

    vid_path = f"{BASE_PATH}/{vid_name}"
    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]

    csv_name = f"{RESULTS_LOC}/{name}_highpass.csv"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = f"{RESULTS_LOC}/{name}_highpass.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # TRACKING DS
    frame_count = 0
    next_object_id = 0
    tracked_objects = {}
    lost_objects = {}
    object_paths = {}
    object_lifetimes = {}
    colors = {}

    object_velocities = {}
    last_centers = {}

    tracking_data = []

    if not cap.isOpened():
        print(f"Error opening {vid_path}")
        continue

    # FRAME
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        continue

    print(f"{name}: 1st frame contour report")
    large_contours, disp_frm, fly_mask, blue, low, hf_vis = segment_frame(frame, verbose=True)

    if SHOW_PREVIEW:
        show_preview(frame, low, hf_vis, fly_mask, disp_frm, name)

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

        large_contours, disp_frm, fly_mask, blue, low, hf_vis = segment_frame(frame2)

        if DEBUG:
            cv2.imwrite(f"{DEBUG_LOC}/{name}_blue_mask.png", blue)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_highfreq.png", hf_vis)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_fly_mask.png", fly_mask)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_disp_frm.png", disp_frm)
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

        new_tracked_objects = {}
        used_current = set()
        matched_tracked_ids = set()

        # 0) move ghost positions for all lost flies
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

                # ID-swap fix: reject the match if another *currently tracked* fly's last position is closer to the detection than obj_id's prediction.
                matched_cx, matched_cy = get_center(matched_bbox)
                pred_cx, pred_cy = predict_position(tracked_objects[obj_id], obj_id)
                dist_to_pred = np.sqrt((matched_cx - pred_cx) ** 2 + (matched_cy - pred_cy) ** 2)

                stolen = False
                for other_id, other_bbox in tracked_objects.items():
                    if other_id == obj_id:
                        continue
                    other_cx, other_cy = get_center(other_bbox)
                    dist_other = np.sqrt((matched_cx - other_cx) ** 2 + (matched_cy - other_cy) ** 2)
                    if dist_other < dist_to_pred * 0.5 and dist_to_pred > DISTANCE_THRESHOLD:
                        stolen = True
                        break

                if stolen:
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

        # 2) recovery w the ghost pos
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
                if obj_id in lost_to_remove:
                    continue

                gc = ghost_center(obj_id)
                if gc is not None:
                    det_cx, det_cy = get_center(curr_bbox)
                    dist = np.sqrt((gc[0] - det_cx) ** 2 + (gc[1] - det_cy) ** 2)
                else:
                    dist = calculate_distance(lost_data['bbox'], curr_bbox)

                if dist < best_lost_dist:
                    best_lost_dist = dist
                    best_lost_id = obj_id

            if best_lost_id != -1:
                fly_was_fast = (np.sqrt(sum(v ** 2 for v in get_velocity(best_lost_id))) > FLYING_SPEED_THRESHOLD)
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

            # truly a new fly
            new_tracked_objects[next_object_id] = curr_bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(curr_bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            object_velocities[next_object_id] = (0.0, 0.0)
            last_centers[next_object_id] = (cx, cy)
            print(f"  [NEW FLY] ID {next_object_id} at ({cx}, {cy})")
            next_object_id += 1

        # 3) Clean up lost buffer
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

        # drawing
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                label = f'ID:{obj_id} [FLY]' if is_flying(obj_id) else f'ID:{obj_id}'
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if DEBUG:
            for obj_id, data in lost_objects.items():
                gc = ghost_center(obj_id)
                if gc is not None:
                    color = get_unique_color(obj_id)
                    cv2.circle(frame2, (int(gc[0]), int(gc[1])), 8, color, 2)
                    cv2.putText(frame2, f'L:{obj_id}', (int(gc[0]) + 10, int(gc[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        valid_flies = sum(1 for oid in tracked_objects if object_lifetimes.get(oid, 0) >= MIN_LIFETIME)
        cv2.putText(frame2, f'TOTAL FLIES: {valid_flies}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        out.write(frame2)

        if frame_count % fps == 0:
            print(f"{name} @ frame {frame_count}| {frame_count // fps} sec: valid={valid_flies},tracked={len(tracked_objects)},lost={len(lost_objects)}")

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