'''
Tutorial used:
https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/

Issues found:
- Flies being redected after inactivity as new flies
- Flies sometimes being changed into new ID flies for seemingly no reason, why is that happening
- Zig-zaggy paths, this is due to the matching algorithm -> make it use velocity to see the changes

Solutions implemented:
- Added path tracking and visualization
- Filter out objects that appear for less than 5 frames
- Improved ID consistency with temporal tracking and lost object recovery
'''

import cv2
import numpy as np
import random
import csv
from collections import deque

for name, min_contour_area in [
                            # ("1x_bettercrop", 5),
                            # ("1x_speed", 10),
                            # ("20x_speed", 5),
                            ("plate_d1", 30),
                            #   ("vial_closeup", 10), 
                            #    ("vial_d3", 10), 
                            #    ("vial_d2", 10), 
                            #    ("vial_d5", 10)
                               ]:
    vid_path = f"SampleVideos/{name}.mp4"
    cap = cv2.VideoCapture(vid_path)
    csv_name = f"Tracked_{name}.csv"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = f'WatershedAlgorithm/{name}_path_written.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    backSub = cv2.createBackgroundSubtractorKNN()

    # Object tracking dictionaries
    next_object_id = 0
    tracked_objects = {}  # key = object_id and value = (x, y, w, h)
    lost_objects = {}  # Temporarily store lost objects for recovery for when flies disappear and then appear again
    object_paths = {}  # key = object_id and value = deque of (cx, cy) positions
    object_lifetimes = {}  # key = object_id and value = number of frames seen
    colors = {}  # key = object_id and value = color
    
    # To store into CSV the different coordinates across frames and fly ID's
    tracking_data = []  # List of dictionaries, one per frame
    
    # Tracking parameters
    MAX_LOST_FRAMES = 10  # how many frames to keep lost objects for potential recovery
    MIN_LIFETIME = 5  # min frames an object must be seen to be considered valid
    MAX_PATH_LENGTH = 50  # max number of points to keep in path history
    DISTANCE_THRESHOLD = 100  # max distance for matching (reduced from 200)

    def get_center(bbox):
        x, y, w, h = bbox
        return (x + w//2, y + h//2)

    def calculate_distance(bbox1, bbox2):
        x1, y1 = get_center(bbox1)
        x2, y2 = get_center(bbox2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
        # Step 1: Noise removal using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Step 2: Sure background area (dilation)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Step 3: Distance transform to find sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Step 4: Threshold to get sure foreground
        # Using 0.5 * max distance as threshold
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Step 5: Find unknown region (neither sure bg nor sure fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Step 6: Marker labeling
        # Label connected components in sure foreground
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so background is not 0, but 1
        markers = markers + 1
        
        # Mark the unknown region with 0
        markers[unknown == 255] = 0
        
        # Step 7: Apply watershed algorithm
        # Need 3-channel image for watershed
        if len(original_frame.shape) == 2:
            original_frame_3ch = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
        else:
            original_frame_3ch = original_frame.copy()
        
        markers = cv2.watershed(original_frame_3ch, markers)
        
        # Step 8: Extract contours from watershed result
        # Get unique labels (excluding -1 which is boundary and 1 which is background)
        labels = np.unique(markers)
        contours_list = []
        
        for label in labels[2:]:  # Skip background (1) and start from 2
            # Create binary mask for this label
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            
            # Find contours in this binary mask
            contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contours_list.append(contours[0])
        
        return contours_list

    def draw_paths(frame, paths, obj_id):
        """Draw the path history for a given object"""
        if obj_id in paths and len(paths[obj_id]) > 1:
            color = get_unique_color(obj_id)
            points = list(paths[obj_id])
            
            # Draw lines connecting the path points
            for i in range(len(points) - 1):
                # Make older points more transparent (fade effect)
                # alpha = (i + 1) / len(points)
                # thickness = max(1, int(2 * alpha))
                thickness = 1
                cv2.line(frame, points[i], points[i + 1], color, thickness)
            
            # Draw a circle at the most recent position
            cv2.circle(frame, points[-1], 3, color, -1)

    if not cap.isOpened():
        print("Error video bad")
        exit()
    else:
        ret, frame1 = cap.read()
        
        # Initialize first frame objects
        fg_mask = backSub.apply(frame1)
        
        # Apply Otsu's thresholding for better binary image
        ret, fg_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply watershed segmentation
        contours = apply_watershed_segmentation(fg_mask, frame1)
        
        # Filter by area
        max_contour_area = 25
        large_contours = [cnt for cnt in contours 
                        if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
        
        # Assign IDs to objects in first frame
        for cnt in large_contours:
            bbox = cv2.boundingRect(cnt)
            tracked_objects[next_object_id] = bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            next_object_id += 1
        
        # Store first frame data
        frame_data = {'frame': 0}
        for obj_id, bbox in tracked_objects.items():
            cx, cy = get_center(bbox)
            frame_data[f'ID_{obj_id}'] = f'({cx},{cy})'
        tracking_data.append(frame_data)
        
        frame_count = 0
        print(f"Starting tracking with {len(tracked_objects)} initial flies detected")
        
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame2 = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Apply background subtraction
            fg_mask = backSub.apply(frame2)
            
            # Apply Otsu's thresholding
            ret, fg_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply watershed segmentation to separate touching flies
            contours = apply_watershed_segmentation(fg_mask, frame2)
            
            # Filter contours by size
            min_contour_area = 25
            max_contour_area = 300
            large_contours = [cnt for cnt in contours 
                            if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
            
            # Get bounding boxes for current frame
            current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]
            
            # Match current objects to tracked objects
            new_tracked_objects = {}
            used_current = set()
            
            # First, try to match with currently tracked objects
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
                
                # Assign match if found and distance is reasonable
                if best_match_i != -1 and min_dist < DISTANCE_THRESHOLD:
                    new_tracked_objects[obj_id] = current_bboxes[best_match_i]
                    used_current.add(best_match_i)
                    object_lifetimes[obj_id] += 1
                    
                    # Update path
                    cx, cy = get_center(current_bboxes[best_match_i])
                    object_paths[obj_id].append((cx, cy))
                else:
                    # Object lost means we add it to lost objects for potential recovery when flies re-appear
                    if obj_id not in lost_objects:
                        lost_objects[obj_id] = {'bbox': prev_bbox, 'frames_lost': 1}
                    else:
                        lost_objects[obj_id]['frames_lost'] += 1
            
            # Try to recover lost objects with remaining unmatched detections
            lost_to_remove = []
            for obj_id, lost_data in lost_objects.items():
                if lost_data['frames_lost'] > MAX_LOST_FRAMES:
                    lost_to_remove.append(obj_id)
                    continue
                
                min_dist = float('inf')
                best_match_i = -1
                
                for i, curr_bbox in enumerate(current_bboxes):
                    if i in used_current:
                        continue
                    dist = calculate_distance(lost_data['bbox'], curr_bbox)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_i = i
                
                # Recover the object if a good match is found
                if best_match_i != -1 and min_dist < DISTANCE_THRESHOLD * 2:  # More lenient for recovery of lost flies
                    new_tracked_objects[obj_id] = current_bboxes[best_match_i]
                    used_current.add(best_match_i)
                    object_lifetimes[obj_id] += 1
                    
                    # Update path
                    cx, cy = get_center(current_bboxes[best_match_i])
                    object_paths[obj_id].append((cx, cy))
                    
                    lost_to_remove.append(obj_id)
            
            # Remove recovered or expired lost objects
            for obj_id in lost_to_remove:
                del lost_objects[obj_id]
            
            # Assign new IDs to remaining unmatched flies
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
            
            # Draw paths and bounding boxes (only for objects with sufficient lifetime)
            for obj_id, bbox in tracked_objects.items():
                if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                    # Draw path
                    draw_paths(frame2, object_paths, obj_id)
                    
                    # Draw bounding box
                    x, y, w, h = bbox
                    color = get_unique_color(obj_id)
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame2, f'ID:{obj_id}', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame2)
            
            if frame_count % 30 == 0:
                valid_flies = sum(1 for obj_id in tracked_objects if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME)
                print(f"{name} @ {frame_count} frames with {valid_flies} valid flies (total tracked: {len(tracked_objects)})")

    # Write tracking data to CSV after processing all frames
    if tracking_data:
        # Get all unique fly IDs that appeared across all frames with sufficient lifetime
        all_fly_ids = set()
        for frame_data in tracking_data:
            all_fly_ids.update([k for k in frame_data.keys() if k.startswith('ID_')])
        
        sorted_fly_ids = sorted(all_fly_ids, key=lambda x: int(x.split('_')[1]))
        
        # Write CSV
        with open(csv_name, 'w', newline='') as csvfile:
            # Create header: Frame, ID_0, ID_1, ID_2, ...
            fieldnames = ['frame'] + sorted_fly_ids
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write each frame's data
            for frame_data in tracking_data:
                # Fill in empty values for flies not present in this frame
                row = {'frame': frame_data['frame']}
                for fly_id in sorted_fly_ids:
                    row[fly_id] = frame_data.get(fly_id, '')
                writer.writerow(row)
        
        # print(f"Tracking data saved to {csv_name}")
        print(f"Total unique flies tracked: {len(sorted_fly_ids)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()