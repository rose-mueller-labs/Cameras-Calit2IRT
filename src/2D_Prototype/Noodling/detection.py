import cv2
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# import time

vid_path="/home/drosophila-lab/Documents/Cameras-Calit2IRT/src/2D_Prototype/Noodling/SampleVideos/plate_d1.mp4"
# print(cv2.getBuildInformation())
cap = cv2.VideoCapture(vid_path)


# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 2. Define the codec and create VideoWriter object
output_path = 'plate_d1_written_hgh_min_cntr.mp4'  # Replace with your desired output file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files (e.g., 'XVID' for .avi)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


subtractor_name=""
backSub = cv2.createBackgroundSubtractorKNN() # cv2.createBackgroundSubtractorMOG2()
prev_frame = None
if not cap.isOpened():
    print("Error opening video file")
    exit()
else:
    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret:
        # Apply background subtraction
        prev_frame = frame
        fg_mask = backSub.apply(frame)
      if not ret:
          break
        
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #   break
      # print(frame)
      # frame = prev_frame
      contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      # print(contours)
      # print(hierarchy)
      frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
      # print(frame_ct)
      # Display the resulting frame
      # cv2.imwrite(f'Frame_final_{subtractor_name}.png', frame_ct)
      # time.sleep(20)

      retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
      # Apply erosion
      mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

      min_contour_area = 25  # Define your minimum area threshold
      large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
      # print(large_contours)
      # frame_out = frame.copy()
      for cnt in large_contours:
          x, y, w, h = cv2.boundingRect(cnt)
          frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
      out.write(frame)
      # Display the resulting frame
      # frame_out.write(frame)
      # cv2.show(f'Frame_final2_{subtractor_name}.png', frame_out)
# cv2.imshow('Vid', frame)
cv2.waitKey()
cap.release()
# out.release() # Uncomment if using VideoWriter
cv2.destroyAllWindows()