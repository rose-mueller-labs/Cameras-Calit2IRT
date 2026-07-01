'''
took the og pathing script and swapped out the detection layer.
now runs two detectors in parallel:
  - MOG2 for moving flies (full frame, masked to inner ROI before watershed)
  - static appearance detector for still flies (inner ROI only, CLAHE + adaptive threshold)
both layers get merged and deduplicated before tracking.
also added adaptive HSV color calibration — samples fly colors from confirmed
MOG2 detections and uses that range to gate the static detector, so background
blobs that pass thresholding but have the wrong color get filtered out.
'''

import cv2
import numpy as np
import random
import csv
import os
from collections import deque

BASE_PATH = "/Volumes/Crucial X9/Downloads/Blue"

save_flies = False
debug_mask = False

# static detector
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
STATIC_BLOCK_SIZE = 31
STATIC_C = 8
STATIC_MIN_AREA = 200
STATIC_MAX_AREA = 800
IOU_MERGE_THRESHOLD = 0.3

# adaptive HSV color calibration
MIN_CALIBRATION_FLIES = 2
COLOR_TOLERANCE_H = 15
COLOR_TOLERANCE_S = 60
COLOR_TOLERANCE_V = 60
COLOR_HISTORY_FRAMES = 30
STOP_SEC = 5

# inner ROI - static detector only fires here, MOG2 still runs full frame
STATIC_BORDER_MARGIN = 40

skip_list = ['fb1.mov', 'fb2.mov', 'fb3.mov', 'fb4.mov', 
             'fm1.mov']

for vid_name in os.listdir(BASE_PATH): # @ fm2
    vid_path = f"{BASE_PATH}/{vid_name}"
    if vid_name.startswith('.'):
        continue

    if vid_name in skip_list:
        continue

    cap = cv2.VideoCapture(vid_path)
    name = vid_name
    csv_name = f"./2D_Detection/WatershedAlgorithm/Output/Pathing/Blue/Tracked_{name}_pws.csv"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = f'./2D_Detection/WatershedAlgorithm/Output/Pathing/Blue/{name}_path_written_watershed.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    mid = total_frames // 2
    snapshot_frames = {mid, mid + 1}

    m = STATIC_BORDER_MARGIN
    inner_roi = (m, m, frame_width - 2*m, frame_height - 2*m)

    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=8, detectShadows=False)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))

    hsv_sample_history = deque(maxlen=COLOR_HISTORY_FRAMES)
    calibrated_hsv_range = [None]  # wrapped in list so inner functions can mutate it

    # tracking state
    next_object_id = 0
    tracked_objects = {}
    lost_objects = {}
    object_paths = {}
    object_lifetimes = {}
    colors = {}
    tracking_data = []

    MAX_LOST_FRAMES = 10
    MIN_LIFETIME = 5
    MAX_PATH_LENGTH = 50
    DISTANCE_THRESHOLD = 50
    RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2
    min_contour_area = 250
    max_contour_area = 800

    # ------------
    def get_center(bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def calculate_distance(bbox1, bbox2):
        x1, y1 = get_center(bbox1)
        x2, y2 = get_center(bbox2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def compute_iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(x1, x2); iy = max(y1, y2)
        iw = min(x1+w1, x2+w2) - ix
        ih = min(y1+h1, y2+h2) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0

    def deduplicate_bboxes(bboxes, iou_thresh=IOU_MERGE_THRESHOLD):
        kept = []
        for bbox in bboxes:
            if not any(compute_iou(bbox, k) >= iou_thresh for k in kept):
                kept.append(bbox)
        return kept

    def get_unique_color(obj_id):
        if obj_id not in colors:
            colors[obj_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return colors[obj_id]

    # ------------
    def sample_hsv_from_mog2_flies(bgr_frame, mog2_bboxes):
        # sample median HSV from each confirmed MOG2 bbox
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        samples = []
        for (x, y, w, h) in mog2_bboxes:
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(bgr_frame.shape[1], x+w)
            y2 = min(bgr_frame.shape[0], y+h)
            roi = hsv_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            samples.append((float(np.median(roi[:, :, 0])),
                             float(np.median(roi[:, :, 1])),
                             float(np.median(roi[:, :, 2]))))
        return samples

    def update_calibrated_range(new_samples):
        # push new samples into rolling history and recompute the HSV gate range
        for s in new_samples:
            hsv_sample_history.append(s)
        if len(hsv_sample_history) < MIN_CALIBRATION_FLIES:
            calibrated_hsv_range[0] = None
            return
        arr = np.array(list(hsv_sample_history))
        med_h = float(np.median(arr[:, 0]))
        med_s = float(np.median(arr[:, 1]))
        med_v = float(np.median(arr[:, 2]))
        calibrated_hsv_range[0] = (
            max(0, med_h - COLOR_TOLERANCE_H), min(180, med_h + COLOR_TOLERANCE_H),
            max(0, med_s - COLOR_TOLERANCE_S), min(255, med_s + COLOR_TOLERANCE_S),
            max(0, med_v - COLOR_TOLERANCE_V), min(255, med_v + COLOR_TOLERANCE_V),
        )

    def apply_color_gate(static_mask, bgr_frame):
        # AND the static mask with an HSV inRange mask from calibrated fly colors
        # if not yet calibrated, pass through unchanged
        if calibrated_hsv_range[0] is None:
            return static_mask
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = calibrated_hsv_range[0]
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(
            hsv,
            np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
            np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        )
        return cv2.bitwise_and(static_mask, color_mask)

    # ------------
    def get_static_detections(gray_frame, bgr_frame):
        # runs inside inner ROI only - CLAHE -> adaptive threshold --> color gate ---> contours
        rx, ry, rw, rh = inner_roi
        gray_roi = gray_frame[ry:ry+rh, rx:rx+rw]
        bgr_roi = bgr_frame[ry:ry+rh, rx:rx+rw]

        enhanced = clahe.apply(gray_roi)
        static_mask = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            STATIC_BLOCK_SIZE, STATIC_C
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = apply_color_gate(cleaned, bgr_roi)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if perim == 0 or area < STATIC_MIN_AREA or area > STATIC_MAX_AREA:
                continue
            circ = (4 * 3.14159 * area) / (perim**2)
            if circ < 0.2 or circ > 0.8:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            bboxes.append((bx + rx, by + ry, bw, bh))  # translate back to full-frame coords

        full_static_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        full_static_mask[ry:ry+rh, rx:rx+rw] = cleaned
        return bboxes, full_static_mask

    # -------------------------------------------------------------------------
    def apply_watershed_segmentation(fg_mask, original_frame):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        frame_3ch = (cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
                     if len(original_frame.shape) == 2 else original_frame.copy())
        markers = cv2.watershed(frame_3ch, markers)
        contours_list = []
        for label in np.unique(markers)[2:]:
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours_list.append(cnts[0])
        return contours_list

    def draw_paths(frame, paths, obj_id):
        if obj_id in paths and len(paths[obj_id]) > 1:
            color = get_unique_color(obj_id)
            points = list(paths[obj_id])
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], color, 2)
            cv2.circle(frame, points[-1], 3, color, -1)

    def save_fly_crops(frame, tracked_objects, object_lifetimes, frame_count, name):
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) < MIN_LIFETIME:
                continue
            x, y, w, h = bbox
            crop = frame[max(0, y):min(frame.shape[0], y+h),
                         max(0, x):min(frame.shape[1], x+w)]
            if crop.size == 0:
                continue
            cv2.imwrite(
                f"./2D_Detection/WatershedAlgorithm/Output/Pathing/Blue/fly_crop_{name}_frame{frame_count}_ID{obj_id}.png", crop
            )

    def get_combined_bboxes(mog2_mask, gray_frame, bgr_frame):
        # layer 1: MOG2 masked to inner ROI --> watershed (moving flies)
        rx, ry, rw, rh = inner_roi
        mog2_inner = np.zeros_like(mog2_mask)
        mog2_inner[ry:ry+rh, rx:rx+rw] = mog2_mask[ry:ry+rh, rx:rx+rw]

        ws_contours = apply_watershed_segmentation(mog2_inner, bgr_frame)
        mog2_bboxes = []
        for c in ws_contours:
            if not (min_contour_area < cv2.contourArea(c) < max_contour_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            cx_b = bx + bw // 2; cy_b = by + bh // 2
            if not (rx <= cx_b <= rx+rw and ry <= cy_b <= ry+rh):
                continue
            mog2_bboxes.append((bx, by, bw, bh))

        # update color calibration from this frame's MOG2 flies
        if len(mog2_bboxes) >= MIN_CALIBRATION_FLIES:
            update_calibrated_range(sample_hsv_from_mog2_flies(bgr_frame, mog2_bboxes))

        # layer 2: static detector (inner ROI only, still flies)
        static_bboxes, static_mask = get_static_detections(gray_frame, bgr_frame)

        all_bboxes = deduplicate_bboxes(mog2_bboxes + static_bboxes)
        return all_bboxes, static_mask, mog2_bboxes

    # ---
    if not cap.isOpened():
        print(f"Error: could not open {vid_path}")
        continue

    print(f"Warming up background model for {name}...")
    for _ in range(30):
        ret, warmup_frame = cap.read()
        if not ret:
            break
        backSub.apply(warmup_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Warmup complete, starting tracking...")

    ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    mog2_mask = backSub.apply(frame1)

    current_bboxes, _, _ = get_combined_bboxes(mog2_mask, gray1, frame1)

    for bbox in current_bboxes:
        tracked_objects[next_object_id] = bbox
        object_lifetimes[next_object_id] = 1
        cx, cy = get_center(bbox)
        object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
        next_object_id += 1

    frame_data = {'frame': 0}
    for obj_id, bbox in tracked_objects.items():
        cx, cy = get_center(bbox)
        frame_data[f'ID_{obj_id}'] = f'({cx},{cy})'
    tracking_data.append(frame_data)

    frame_count = 0
    print(f"Starting tracking with {len(tracked_objects)} initial flies detected")

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret or frame_count >= fps * STOP_SEC:
            break

        frame_count += 1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        mog2_mask = backSub.apply(frame2)

        current_bboxes, static_mask, mog2_bboxes = get_combined_bboxes(mog2_mask, gray2, frame2)

        if debug_mask:
            combined_vis = np.zeros((frame_height, frame_width), dtype=np.uint8)
            for bbox in current_bboxes:
                x, y, w, h = bbox
                cv2.rectangle(combined_vis, (x, y), (x+w, y+h), 255, -1)
            rx, ry, rw, rh = inner_roi
            cv2.rectangle(combined_vis, (rx, ry), (rx+rw, ry+rh), 128, 1)
            cal_str = (f"HSV cal: H={calibrated_hsv_range[0][0]:.0f}-{calibrated_hsv_range[0][1]:.0f}"
                       if calibrated_hsv_range[0] else "HSV cal: uncalibrated")
            cv2.putText(combined_vis, cal_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
            cv2.imshow("1 - MOG2 mask", mog2_mask)
            cv2.imshow("2 - Static mask", static_mask)
            cv2.imshow("3 - Combined bboxes", combined_vis)
            cv2.waitKey(1)

        new_tracked_objects = {}
        used_current = set()

        # match detections to active tracks
        for obj_id, prev_bbox in tracked_objects.items():
            if not current_bboxes:
                break
            min_dist, best_i = float('inf'), -1
            for i, curr_bbox in enumerate(current_bboxes):
                if i in used_current:
                    continue
                d = calculate_distance(prev_bbox, curr_bbox)
                if d < min_dist:
                    min_dist, best_i = d, i
            if best_i != -1 and min_dist < DISTANCE_THRESHOLD:
                new_tracked_objects[obj_id] = current_bboxes[best_i]
                used_current.add(best_i)
                object_lifetimes[obj_id] += 1
                cx, cy = get_center(current_bboxes[best_i])
                object_paths[obj_id].append((cx, cy))
            else:
                if obj_id not in lost_objects:
                    lost_objects[obj_id] = {'bbox': prev_bbox, 'frames_lost': 1}
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

        # try to recover lost flies before spawning new IDs
        lost_to_remove = []
        for i, curr_bbox in enumerate(current_bboxes):
            if i in used_current:
                continue
            best_lost_id, best_lost_dist = -1, float('inf')
            for obj_id, lost_data in lost_objects.items():
                if lost_data['frames_lost'] > MAX_LOST_FRAMES:
                    continue
                if obj_id in new_tracked_objects:
                    continue
                d = calculate_distance(lost_data['bbox'], curr_bbox)
                if d < best_lost_dist:
                    best_lost_dist, best_lost_id = d, obj_id
            if best_lost_id != -1 and best_lost_dist < RECOVERY_THRESHOLD:
                new_tracked_objects[best_lost_id] = curr_bbox
                used_current.add(i)
                object_lifetimes[best_lost_id] += 1
                cx, cy = get_center(curr_bbox)
                object_paths[best_lost_id].append((cx, cy))
                lost_to_remove.append(best_lost_id)
                print(f"  [RECOVERY] fly ID {best_lost_id} recovered at "
                      f"{best_lost_dist:.1f}px after "
                      f"{lost_objects[best_lost_id]['frames_lost']} lost frames")
            else:
                new_tracked_objects[next_object_id] = curr_bbox
                object_lifetimes[next_object_id] = 1
                cx, cy = get_center(curr_bbox)
                object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
                next_object_id += 1

        for obj_id in lost_to_remove:
            del lost_objects[obj_id]
        for obj_id in [k for k, v in lost_objects.items() if v['frames_lost'] > MAX_LOST_FRAMES]:
            del lost_objects[obj_id]

        tracked_objects = new_tracked_objects

        frame_data = {'frame': frame_count}
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                cx, cy = get_center(bbox)
                frame_data[f'ID_{obj_id}'] = f'({cx},{cy})'
        tracking_data.append(frame_data)

        if save_flies and frame_count in snapshot_frames:
            save_fly_crops(frame2, tracked_objects, object_lifetimes, frame_count, name)

        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame2, f'ID:{obj_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        rx, ry, rw, rh = inner_roi
        cv2.rectangle(frame2, (rx, ry), (rx+rw, ry+rh), (200, 200, 200), 1)
        cv2.putText(frame2, f'TOTAL FLIES: {len(tracked_objects)}',
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        out.write(frame2)

        if frame_count % 30 == 0:
            valid = sum(1 for o in tracked_objects if object_lifetimes.get(o, 0) >= MIN_LIFETIME)
            cal_status = f"cal={calibrated_hsv_range[0] is not None}"
            print(f"{name} @ {frame_count} frames | valid: {valid} | "
                  f"total: {len(tracked_objects)} | lost: {len(lost_objects)} | {cal_status}")

    if tracking_data:
        all_ids = sorted(
            {k for fd in tracking_data for k in fd if k.startswith('ID_')},
            key=lambda x: int(x.split('_')[1])
        )
        with open(csv_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame'] + all_ids)
            writer.writeheader()
            for fd in tracking_data:
                writer.writerow({'frame': fd['frame'],
                                 **{fid: fd.get(fid, '') for fid in all_ids}})
        print(f"Total unique flies tracked: {len(all_ids)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Saved to {output_path}')