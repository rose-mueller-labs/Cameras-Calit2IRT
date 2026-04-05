import cv2 

vid = cv2.VideoCapture("SampleVideos/2k 120fps backlit.MXF")

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV at ({x},{y}): {hsv[y,x]}")

count, success = 0, True
while success:
    success, frame = vid.read() # Read frame
    if success: 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("frame", frame)
        cv2.setMouseCallback("frame", on_click)
        cv2.waitKey(0)

vid.release()