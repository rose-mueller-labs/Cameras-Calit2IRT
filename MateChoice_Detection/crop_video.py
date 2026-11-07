import cv2
import matplotlib.pyplot as plt

RESULTS_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/MateChoice_Detection/TestingSet/CroppedSet"

cap = cv2.VideoCapture("/Volumes/Crucial X9/Cameras-Calit2IRT/MateChoice_Detection/TestingSet/matechoice-ACO1-test2.mp4")

frame_width = int(2650-1444)
frame_height = int(1420-550)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = f"{RESULTS_LOC}/CROPPED_matechoice-ACO1-test2.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = frame[550:1420, 1444:2650]
        out.write(frame2)
        # print("wrtten")
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()
cap.release()
out.release()