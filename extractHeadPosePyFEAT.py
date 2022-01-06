from feat import Detector
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
import cv2
import pandas as pd
from scipy import stats
import numpy as np
import os
import math
import sys
import cv2
import scipy
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
def getPoint(ex,n):
    x_label = 'x_' + str(n)
    y_label = 'y_' + str(n)
    x = int(ex[x_label][0])
    y = int(ex[y_label][0
    ])
    return (x,y)

def estimateHeadPose(shape,size):
    image_rows, image_cols, _ = size

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
             [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype = "double")
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

    image_points = np.array([
                            getPoint(shape,30),       # Nose tip
                            getPoint(shape,8),        # Chin
                            getPoint(shape,36),       # Left eye left corner
                            getPoint(shape,45),       # Right eye right corne
                            getPoint(shape,48),       # Left Mouth corner
                            getPoint(shape,54),       # Right mouth corner
                        ], dtype="double")
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # Rotation from r
    r = Rotation.from_rotvec(rotation_vector.reshape(3,))
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    angles = r.as_euler('zxy',degrees=True)

    x_angle = int(angles[1])
    y_angle  = int(angles[2])
    z_angle = int(angles[0])
    return x_angle,y_angle,z_angle

while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fname = 'img.jpg'
    cv2.imwrite(fname,image)
    pre = detector.detect_image(fname)
    x,y,z = estimateHeadPose(pre,image.shape)
    #print(head)
    cv2.putText(image, 'X:'+str(x), (10,60), font, 2, (255, 0, 128), 3)
    cv2.putText(image, 'Y:'+str(y), (10,110), font, 2, (255, 0, 128), 3)
    cv2.putText(image, 'Z:'+str(z), (10,170), font, 2, (255, 0, 128), 3)
    cv2.imshow('preview',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
