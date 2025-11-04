# Cameras-Calit2IRT

## General Timeline of the Project
_Fall 2025 (2D prototype)_: Recording, training, testing approaches on one to two flies within a petri dish. Have a 2D camera accurately identifying one to two flies and achieve both mating choice and startle response testing for assays for two flies.

_Winter 2025 (3D setup)_: Have a 3D camera setup working, track multiple flies; quantify the startle response testing and begin behavioral (mating assay) classification. 

_Spring 2025 (Analysis)_: Run long-term group assays; finalize data analysis and presentation.

## Fall 2025 Plan
- Made a recording based on sample `plate_d1.mp4` that detects some of the flies using KNN BackgroundSubtractor.

### 2D Algorithm Plan:
1. Fix the `[50] frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)` so that the countours can be differentiated at the later step. Each fly's contours draw should NOT be touching so that flies can be differentiated at the bounding step.
2. Change `[67] frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)` to drawing a tightly bounded ellipse around the fly instead.
3. Extract coordinates of the detected fly and compare to previous prediction in last iteration of frame to get fly ID
4. Be able to predict the next possible coordinate of the fly and assume that fly in the next frame to be the same fly.
   (a) Some rule-based method, like the velocity thing
   (b) Monte Carlo method of motion prediction
   (c) Another model trained on the flight of one fly used to predict the flight path of current bound-boxed fly.
5. Save the prediction to be used 
6. Write into CSV file the fly's coordinate under it's fly ID (column) for this CURRENT frame (row)
7. Go onto next iteration of frame.

### 2D Algorithm Pseudo-code
```
frames = read_video(path)
backSub = createBackgroundSubtractorKNN()
fly_predictions_prev = dict(key: coordinate, value: fly_id)
while frames:
   frame = cap.read()
   fg_mask = backSub.apply(frame) # apply background subtractor
   contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   frame_contoured = draw_contours(frame, contours) # draw contours that are tightly bounded and leave space between flies near each other
   retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY) # thresholding
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel) # apply erosion
   relevant_contours = [contour.area > 10 for contour in contours] # retrieve contours that are bigger than a minimum contour area
   fly_prediction = dict(key: coordinate, value: fly_id)
   for cnt in relevant_contours:
      x, y, ... = bounding_ellipse(cnt)
      frame = cv2.bounding_ellipse(frame, (x, y), ..., (0, 0, 200), 3)
      coordinate = get_current_coordinate(x, y, ...) # get current coordinate
      fly_id, frame = fly_predictions_prev[coordinate] # or coordinate that is closest to it
      # write the fly_id and coordinate into csv's df[frame][fly_id]
      predicted_coordinate = get_next_coordinate(x, y, ...) # method of getting where the fly will go next
      fly_prediction[coordinate] = fly_id
   fly_predictions_prev = fly_prediction # because our current frame will become the previous frame later on

release resources
```
   


