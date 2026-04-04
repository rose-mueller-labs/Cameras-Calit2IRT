import cv2 

frame = cv2.imread("frame.png")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV at ({x},{y}): {hsv[y,x]}")

cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", on_click)
cv2.waitKey(0)