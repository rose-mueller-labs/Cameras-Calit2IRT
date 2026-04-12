'''
Tutorial used:
https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/

Issues found:
- Flies being redetected after inactivity as new flies
- Flies sometimes being changed into new ID flies for seemingly no reason, why is that happening
- Zig-zaggy paths, this is due to the matching algorithm -> make it use velocity to see the changes

Solutions implemented:
- Added path tracking and visualization
- Filter out objects that appear for less than 5 frames
- Improved ID consistency with temporal tracking and lost object recovery
- Hungarian algorithm for optimal global assignment (prevents ID swaps on brush-by)
- Velocity-predicted matching (predicts where each fly will be, not just where it was)
- Stationary fly detection with extended lost-object window (prevents re-ID after rest)
- Replaced KNN background subtraction with HSV color thresholding for dark flies on light bg
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

BASE_PATH="/Volumes/Crucial X9/Cameras-Calit2IRT/src/SampleVideos"

for vid_path, min_contour_area in [
        # (f"{BASE_PATH}/2k 120fps backlit.MXF", 30),
        # (f"{BASE_PATH}/4k 24fps.MXF", 30),
        (f"{BASE_PATH}/4k 60fps.MXF", 30), # src/SampleVideos/4k 60fps.MXF
        (f"{BASE_PATH}/120fps 2K.MXF", 30),
        (f"{BASE_PATH}/180fps 2K.MXF", 30),
        (f"{BASE_PATH}/180fps more flys.MXF", 30)
        ]:
    name = vid_path.split('/')[-1]
    vid_path = f"SampleVideos/{name}"
    cap = cv2.VideoCapture(vid_path)
    csv_name = f"./2D_Detection/WatershedAlgorithm/Output/Backlit/Tracked_{name}_pwsBacklit.csv"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = f'./2D_Detection/WatershedAlgorithm/Output/Backlit/{name}_path_written_watershed_thresh.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # No more KNN background subtractor using HSV thresholding instead

    # Object tracking dictionaries
    next_object_id = 0
    tracked_objects = {} # key = object_id, value = (x, y, w, h)
    lost_objects = {} # Temporarily store lost objects for recovery
    object_paths = {} # key = object_id, value = deque of (cx, cy) positions
    object_lifetimes = {} # key = object_id, value = number of frames seen
    object_was_stationary = {}  # key = object_id, value = bool (was stationary when lost)
    colors = {} # key = object_id, value = color

    # To store into CSV the different coordinates across frames and fly IDs
    tracking_data = []  # List of dictionaries, one per frame

    # Tracking parameters
    MAX_LOST_FRAMES = 10 # how many frames to keep a moving lost object
    STATIONARY_LOST_MULTIPLIER = 5 # multiply MAX_LOST_FRAMES for stationary flies
    MIN_LIFETIME = 5 # min frames an object must be seen to be considered valid
    MAX_PATH_LENGTH = 50 # max number of points to keep in path history
    DISTANCE_THRESHOLD = 100 # max distance for matching
    STATIONARY_THRESHOLD = 3.0 # pixel displacement below which a fly is "stationary"
    VELOCITY_HISTORY = 5 # number of recent frames used to compute velocity

    CURRENT_TOTAL_FLIES = 0

    # HSV at (906,476): [ 19 111 166] fly
    # HSV at (913,474): [ 24 115 146]
    # HSV at (907,477): [ 17 110 171]
    # HSV at (903,482): [ 30 221 111]
    # HSV at (922,466): [105  11 238]
    # HSV at (827,481): [ 30  98 130]
    # HSV at (827,488): [ 31  94 189]
    # HSV at (823,492): [ 38  95 145]
    # HSV at (826,515): [ 31  84 161]
    # HSV at (831,515): [ 31  57 198]
    # HSV at (731,652): [ 35 191 107]
    # HSV at (738,658): [ 23 131 148]
    # HSV at (744,662): [ 36 145 120] fly
    # HSV at (1096,711): [105   2 255] bg
    LOWER_BROWN = np.array([0,  50,  100])
    UPPER_BROWN = np.array([40, 135, 200])

    def get_fly_mask(frame):
        """
        Threshold for dark brown/black flies on a light background.
        Works in HSV for better color separation than raw BGR.
        Replaces KNN background subtraction entirely — stationary flies
        will NOT be absorbed into the background model.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_brown = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
        fg_mask = mask_brown

        # Clean up noise: open removes small specks, close fills small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return fg_mask

    def get_center(bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def calculate_distance(bbox1, bbox2):
        x1, y1 = get_center(bbox1)
        x2, y2 = get_center(bbox2)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Velocity helpers
    def get_velocity(obj_id):
        """Return smoothed (vx, vy) from the last VELOCITY_HISTORY path points."""
        if obj_id not in object_paths or len(object_paths[obj_id]) < 2:
            return (0.0, 0.0)
        points = list(object_paths[obj_id])
        n = min(VELOCITY_HISTORY, len(points) - 1)
        recent = points[-(n + 1):]
        vx = (recent[-1][0] - recent[0][0]) / n
        vy = (recent[-1][1] - recent[0][1]) / n
        return (vx, vy)

    def get_predicted_center(obj_id, bbox):
        """Predict where the fly will be next frame using its current velocity."""
        vx, vy = get_velocity(obj_id)
        cx, cy = get_center(bbox)
        return (cx + vx, cy + vy)

    def distance_to_predicted(predicted_center, curr_bbox):
        cx, cy = get_center(curr_bbox)
        px, py = predicted_center
        return np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

    # Stationary detection
    def is_stationary(obj_id):
        """True if the fly has barely moved over the last VELOCITY_HISTORY frames."""
        if obj_id not in object_paths or len(object_paths[obj_id]) < VELOCITY_HISTORY:
            return False
        points = list(object_paths[obj_id])[-VELOCITY_HISTORY:]
        max_disp = max(
            np.sqrt((p[0] - points[0][0]) ** 2 + (p[1] - points[0][1]) ** 2)
            for p in points
        )
        return max_disp < STATIONARY_THRESHOLD

    # Hungarian-algorithm matching
    def match_objects_hungarian(tracked_objs, current_bboxes, threshold):
        """
        Globally optimal assignment using the Hungarian algorithm.
        Costs are based on velocity-predicted positions.

        Returns:
            matched          – list of (obj_id, bbox_index) pairs
            unmatched_objs   – list of obj_ids with no good match
            unmatched_bboxes – list of bbox indices with no assignment
        """
        if not tracked_objs or not current_bboxes:
            return [], list(tracked_objs.keys()), list(range(len(current_bboxes)))

        obj_ids = list(tracked_objs.keys())

        # Build cost matrix: rows = tracked objects, cols = current detections
        cost_matrix = np.zeros((len(obj_ids), len(current_bboxes)), dtype=np.float32)
        for i, obj_id in enumerate(obj_ids):
            predicted = get_predicted_center(obj_id, tracked_objs[obj_id])
            for j, curr_bbox in enumerate(current_bboxes):
                cost_matrix[i, j] = distance_to_predicted(predicted, curr_bbox)

        # Solve
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_bboxes = set(range(len(current_bboxes)))
        matched_obj_set = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < threshold:
                matched.append((obj_ids[r], c))
                unmatched_bboxes.discard(c)
                matched_obj_set.add(obj_ids[r])

        unmatched_objs = [oid for oid in obj_ids if oid not in matched_obj_set]

        return matched, unmatched_objs, list(unmatched_bboxes)

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

    def apply_watershed_segmentation(fg_mask, original_frame):
        """
        https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/
        1. Noise removal with morphological operations
        2. Sure background detection (dilation)
        3. Distance transform for sure foreground
        4. Unknown region calculation
        5. Marker labeling
        6. Watershed application
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
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
            contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contours_list.append(contours[0])

        return contours_list

    def draw_paths(frame, paths, obj_id):
        """Draw the path history for a given object."""
        if obj_id in paths and len(paths[obj_id]) > 1:
            color = get_unique_color(obj_id)
            points = list(paths[obj_id])
            for i in range(len(points) - 1):
                thickness = 2
                cv2.line(frame, points[i], points[i + 1], color, thickness)
            cv2.circle(frame, points[-1], 3, color, -1)

    if not cap.isOpened():
        print("Error video bad")
        exit()
    else:
        ret, frame1 = cap.read()

        # Initialize first frame objects
        fg_mask = get_fly_mask(frame1)  #  was: backSub.apply + Otsu threshold
        contours = apply_watershed_segmentation(fg_mask, frame1)

        max_contour_area = 25
        large_contours = [cnt for cnt in contours
                          if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

        for cnt in large_contours:
            bbox = cv2.boundingRect(cnt)
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
            if not ret:
                break

            frame_count += 1

            # HSV thresholding instead of KNN background subtraction
            fg_mask = get_fly_mask(frame2)

            # Watershed segmentation
            contours = apply_watershed_segmentation(fg_mask, frame2)

            min_contour_area = 25
            max_contour_area = 300
            large_contours = [cnt for cnt in contours
                              if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

            current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

            new_tracked_objects = {}

            # Hungarian matching on currently tracked objects
            matched, unmatched_obj_ids, unmatched_bbox_indices = match_objects_hungarian(
                tracked_objects, current_bboxes, DISTANCE_THRESHOLD
            )

            # Process successful matches
            for obj_id, bbox_idx in matched:
                new_tracked_objects[obj_id] = current_bboxes[bbox_idx]
                object_lifetimes[obj_id] = object_lifetimes.get(obj_id, 0) + 1
                cx, cy = get_center(current_bboxes[bbox_idx])
                object_paths[obj_id].append((cx, cy))

            used_current = {bbox_idx for _, bbox_idx in matched}

            # Unmatched tracked objects → move to lost_objects
            for obj_id in unmatched_obj_ids:
                # Record whether this fly was stationary before losing it
                stationary = is_stationary(obj_id)
                object_was_stationary[obj_id] = stationary
                extended_limit = (MAX_LOST_FRAMES * STATIONARY_LOST_MULTIPLIER
                                  if stationary else MAX_LOST_FRAMES)

                if obj_id not in lost_objects:
                    lost_objects[obj_id] = {
                        'bbox': tracked_objects[obj_id],
                        'frames_lost': 1,
                        'extended_limit': extended_limit,
                    }
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

            # Try to recover lost objects with remaining detections
            remaining_bbox_indices = [i for i in unmatched_bbox_indices if i not in used_current]
            lost_to_remove = []

            if lost_objects and remaining_bbox_indices:
                lost_ids = list(lost_objects.keys())
                remaining_bboxes = [current_bboxes[i] for i in remaining_bbox_indices]

                cost_matrix_lost = np.zeros((len(lost_ids), len(remaining_bboxes)), dtype=np.float32)
                for i, obj_id in enumerate(lost_ids):
                    predicted = get_predicted_center(obj_id, lost_objects[obj_id]['bbox'])
                    for j, curr_bbox in enumerate(remaining_bboxes):
                        cost_matrix_lost[i, j] = distance_to_predicted(predicted, curr_bbox)

                row_ind, col_ind = linear_sum_assignment(cost_matrix_lost)

                recovery_threshold = DISTANCE_THRESHOLD * 2  # more lenient for re-appearing flies
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix_lost[r, c] < recovery_threshold:
                        obj_id = lost_ids[r]
                        orig_idx = remaining_bbox_indices[c]
                        new_tracked_objects[obj_id] = current_bboxes[orig_idx]
                        object_lifetimes[obj_id] = object_lifetimes.get(obj_id, 0) + 1
                        cx, cy = get_center(current_bboxes[orig_idx])
                        object_paths[obj_id].append((cx, cy))
                        used_current.add(orig_idx)
                        lost_to_remove.append(obj_id)

            # Expire lost objects that have been gone too long
            for obj_id, lost_data in lost_objects.items():
                if obj_id in lost_to_remove:
                    continue
                limit = lost_data.get('extended_limit', MAX_LOST_FRAMES)
                if lost_data['frames_lost'] > limit:
                    lost_to_remove.append(obj_id)
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

            for obj_id in lost_to_remove:
                lost_objects.pop(obj_id, None)
                object_was_stationary.pop(obj_id, None)

            # ----------------------------------------------------------
            # Step 3: Assign new IDs to truly unmatched detections
            # ----------------------------------------------------------
            for i, curr_bbox in enumerate(current_bboxes):
                if i not in used_current:
                    new_tracked_objects[next_object_id] = curr_bbox
                    object_lifetimes[next_object_id] = 1
                    cx, cy = get_center(curr_bbox)
                    object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
                    next_object_id += 1

            tracked_objects = new_tracked_objects

            # Store current frame data (only for objects with sufficient lifetime)
            frame_data = {'frame': frame_count}
            for obj_id, bbox in tracked_objects.items():
                if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                    cx, cy = get_center(bbox)
                    frame_data[f'ID_{obj_id}'] = f'({cx},{cy})'
            tracking_data.append(frame_data)

            # Draw paths and bounding boxes
            CURRENT_TOTAL_FLIES = len(tracked_objects)
            for obj_id, bbox in tracked_objects.items():
                if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                    draw_paths(frame2, object_paths, obj_id)
                    x, y, w, h = bbox
                    color = get_unique_color(obj_id)
                    frame2 = cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frame2, f'ID:{obj_id}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame2, f'TOTAL FLIES: {CURRENT_TOTAL_FLIES}',
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

            out.write(frame2)

            if frame_count % 30 == 0:
                valid_flies = sum(1 for obj_id in tracked_objects
                                  if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
                print(f"{name} @ {frame_count} frames with {valid_flies} valid flies "
                      f"(total tracked: {len(tracked_objects)})")

    # Write tracking data to CSV
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
    print(f'Saved to {output_path}')