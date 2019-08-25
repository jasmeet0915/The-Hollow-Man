import cv2
import numpy as np
import dlib
import time


def shape_to_numpy(shape):
    points = np.zeros((68, 2), 'int32')
    for i in range(0, 68):
        points[i] = (shape.part(i).x, shape.part(i).y)
    return points


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
count = 0
background = 0
time.sleep(0.5)

while True:
    if count == 0:
        for i in range(60):
            _, background = cap.read()
            background = cv2.resize(background, (background.shape[1]//2, background.shape[0]//2))
            background = cv2.flip(background, 0)
        cv2.imshow("background", background)
        count = 1

    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame = cv2.flip(frame, 0)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(frame.shape, np.uint8)
    mask.fill(0)

    rects = detector(frame_gray, 1)

    for rect in rects:
        landmarks = predictor(frame_gray, rect)
        landmarks = shape_to_numpy(landmarks)
        convex_hull = cv2.convexHull(landmarks)
        #cv2.polylines(frame, [convex_hull], True, (0, 0, 255), 3)
        cv2.fillConvexPoly(mask, convex_hull, (255, 255, 255))

    mask_inv = cv2.bitwise_not(mask)
    final_fg = cv2.bitwise_and(frame, mask_inv)
    final_bg = cv2.bitwise_and(background, mask)
    final = cv2.add(final_bg, final_fg)
    cv2.imshow("final", final)
    cv2.imshow("background", background)
    cv2.imshow("foreground", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()