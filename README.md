# Cameras-Calit2IRT

## General Timeline of the Project
_Fall 2025 (2D prototype)_: Recording, training, testing approaches on one to two flies within a petri dish. Have a 2D camera accurately identifying one to two flies and achieve both mating choice and startle response testing for assays for two flies.

_Winter 2025 (3D setup)_: Have a 3D camera setup working, track multiple flies; quantify the startle response testing and begin behavioral (mating assay) classification. 

_Spring 2025 (Analysis)_: Run long-term group assays; finalize data analysis and presentation.

## Fall 2025 Plan
- Made a recording based on sample `plate_d1.mp4` that detects some of the flies using KNN BackgroundSubtractor.

2D Algorithm Plan:
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


