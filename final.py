# Import libraries
import numpy as np
import cv2
import time
from tensorflow import keras
import copy

# Load Pretrained Model from HandGesture_Model.py
model = keras.models.load_model('saved_model/my_model')

# Initialize Parameters
bg_threshold = 50
learning_rate = 0
blur_value = 41
threshold = 60
cap_region_x_begin = 0.5
cap_region_y_end = 0.8


bgModel = cv2.createBackgroundSubtractorMOG2(0, bg_threshold)


def remove_background(frame):    # Background Removal
    fgmask = bgModel.apply(frame, learningRate=learning_rate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


print('camera opens in 2 secs')         #Camera Alerts
time.sleep(10)
print('opening camera')

camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    global thresh
    global drawing
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)          # smoothing filter
    frame = cv2.flip(frame, 1)                              # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    img = remove_background(frame)
    img = img[0:int(cap_region_y_end*frame.shape[0]), int(cap_region_x_begin*frame.shape[1]):frame.shape[1]]
    cv2.imshow('mask', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('threshold', thresh)
    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):              # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    cv2.imshow('output', drawing)

    key = cv2.waitKey(10)
    if key == 27:
        break

drawing = cv2.resize(drawing, (128, 128))
a = np.array([drawing])

camera.release()
cv2.destroyAllWindows()

pred = model.predict(a)
print(pred)
final_pred = np.argmax(pred, axis = 1)
print(final_pred, type(final_pred))
final_pred = int(final_pred)

classification = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved','c','down']
print(classification[final_pred])