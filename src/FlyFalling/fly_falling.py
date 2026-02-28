import cv2
import matplotlib.pyplot as plt
import numpy as np

def debug(frame):
    rgimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgimg)
    plt.show()

def average_rgb(frame):
    average_color_row = np.average(frame, axis=0)
    average_color = np.average(average_color_row, axis=0)
    return average_color

name = 'Recording from 2026-02-11 13-11-28.511852.webm'
vid_path = f"SampleVideos/{name}"
cap = cv2.VideoCapture(vid_path)
csv_name = f"FlyFalling/Tracked_{name}.csv"

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = f'{name}_written.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 3. open the csv file which will contain coors (intermediate)
# |timestamp | timestamp ...
# -----------------------...
# |bb coor   | bb coor   ...
# -----------------------...
# with open(f"{name}_written_coordinates.csv", "w", newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Timestamp{i}' * len()])

subtractor_name=""
prev_frame = None
if not cap.isOpened():
    print("Error opening video file")
    exit()
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            prev_frame = frame
        
        if not ret:
            break
        rgb = average_rgb(frame)
        print(rgb)
        
        fg_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        debug(fg_mask)

        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        debug(fg_mask)

        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        retval, mask_thresh = cv2.threshold( fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        min_contour_area = 10  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

        out.write(frame)
      
cv2.waitKey()
cap.release()
out.release()
cv2.destroyAllWindows()