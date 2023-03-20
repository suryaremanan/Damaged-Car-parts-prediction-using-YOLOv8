import cv2
from ultralytics import YOLO
img_pth = "1.jpeg"
model = YOLO("best.pt") 
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
