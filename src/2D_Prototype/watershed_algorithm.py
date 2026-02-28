'''
Tutorial used:
https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/

Issues found:
- Flies being redected after inactivity as new flies
- Flies sometimes being changed into new ID flies for seemingly no reason, why is that happening
'''


import cv2
import numpy as np
import random
import csv

for name, min_contour_area in [
                            # ("1x_bettercrop", 5),
                            # ("plate_d1", 15),
                            #    ("vial_closeup", 10), 
                            #    ("vial_d3", 10), 
                            #    ("vial_d2", 10), 
                            #    ("vial_d5", 10)
                            # ("120fps 2K.MXF", 30)
                            # ("4k 24fps.MXF", 30)
                            ("4k 60fps.MXF", 30)
                               ]:
    vid_path = f"SampleVideos/{name}"
    cap = cv2.VideoCapture(vid_path)
    csv_name = f"Tracked_{name}.csv"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = f'WatershedAlgorithm/{name}_written_watershed.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    backSub = cv2.createBackgroundSubtractorKNN()

    # Object tracking dictionaries
    next_object_id = 0
    tracked_objects = {}  # key = object_id and value = (x, y, w, h)
    colors = {}  # key = object_id and value = color

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
            next_object_id += 1
        
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
            
            # For each tracked object, find closest match in current frame
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
                if best_match_i != -1 and min_dist < 200: # increase min dist cuz it hella making new flies
                    new_tracked_objects[obj_id] = current_bboxes[best_match_i]
                    used_current.add(best_match_i)
            
            # Assign new IDs to unmatched flies
            for i, curr_bbox in enumerate(current_bboxes):
                if i not in used_current:
                    new_tracked_objects[next_object_id] = curr_bbox
                    next_object_id += 1
            
            tracked_objects = new_tracked_objects
            
            # Draw bounding boxes with unique colors for each fly ID
            for obj_id, bbox in tracked_objects.items():
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame2, f'ID:{obj_id}', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame2)
            print(tracked_objects)
            # with open(f"{csv_name}", 'w'):
            #     csv_writer = csv.writer()
                
                
            if frame_count % 30 == 0:
                print(f"{name} @ {frame_count} frames with {len(tracked_objects)} flies")


    cap.release()
    out.release()
    cv2.destroyAllWindows()