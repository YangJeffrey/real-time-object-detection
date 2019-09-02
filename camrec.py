import cv2 as cv

winName = 'Window'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000,1000)

cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:

    hasFrame, frame = cap.read()

    cv.imshow(winName, frame)
