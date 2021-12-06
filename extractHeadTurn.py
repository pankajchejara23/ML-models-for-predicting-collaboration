import cv2
import sys
import mediapipe as mp
import math
import pandas as pd
from typing import List, Mapping, Optional, Tuple, Union
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# this threshold will determine whether we the detected landmark is considered present or not.
# idea is taken from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

# this function is also taken from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
# As MediaPipe return landmarks in normalized form, we need to convert them back relative to the image.
def _normalized_to_pixel_coordinates(
    normalized, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

    if not (is_valid_normalized_value(normalized.x) and
          is_valid_normalized_value(normalized.y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized.x * image_width), image_width - 1)
    y_px = min(math.floor(normalized.y * image_height), image_height - 1)
    z_px = min(math.floor(normalized.z * image_height), image_height - 1)
    return x_px, y_px

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    front_size = img.shape[1]/4
    front_depth = img.shape[1]/4
    rear_size = img.shape[1]/6
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))


    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))


    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    k = (point_2d[5] + point_2d[8])//2
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return(point_2d[2], k)

def extractMediaPipeFeatures(video_file,draw=False,save=False):
    cap = cv2.VideoCapture(0)
    cv2.startWindowThread()
    cv2.namedWindow("preview")

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])




    # columns of the output csv file
    face_landmarks_columns = []
    # there are 468 face landmarks  returned by MediaPipe
    for i in range(468):
        x = str(i)+'_x'
        y = str(i) + '_y'
        face_landmarks_columns.append(x)
        face_landmarks_columns.append(y)
    output_df = pd.DataFrame(columns=face_landmarks_columns)
    # for labeling resultant image
    i = 0
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            if i == 0:
                size = image.shape

                # Camera internals
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image1)

            image_rows, image_cols, _ = image.shape


            if results.face_landmarks:
                shape = results.face_landmarks.landmark
                image_points = np.array([
                                        _normalized_to_pixel_coordinates(shape[1], image_cols, image_rows),     # Nose tip
                                        _normalized_to_pixel_coordinates(shape[152], image_cols, image_rows),     # Chin
                                        _normalized_to_pixel_coordinates(shape[226], image_cols, image_rows),     # Left eye left corner
                                        _normalized_to_pixel_coordinates(shape[446], image_cols, image_rows),     # Right eye right corne
                                        _normalized_to_pixel_coordinates(shape[57], image_cols, image_rows),     # Left Mouth corner
                                        _normalized_to_pixel_coordinates(shape[287], image_cols, image_rows)      # Right mouth corner
                                    ], dtype="double")
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = draw_annotation_box(image1, rotation_vector, translation_vector, camera_matrix)

                #cv2.line(image1, p1, p2, (0, 255, 255), 2)
                #cv2.line(image1, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in shape:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90

                # print('div by zero error')
                cv2.putText(image, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                cv2.putText(image, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)


                # we need image height and width for converting normalized coordinates back to original
                print(p1,p2)
                # Draw landmark annotation on the image.
                image.flags.writeable = True
                cv2.line(image, p1, p2, (0, 255, 255), 2)

                cv2.imshow('preview',image1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            i = i + 1

    cap.release()
    return output_df


if len(sys.argv) < 3:
    print('Incorrect use, please use the script in the following way.\n')
    print("Use this format: \n python extractFaceLandmarks.py <video_file> <feature_file_name> <draw> <save>")
    print('\n      here \n        <video_file> is the name of the video file with the path')
    print('        <feature_file_name> is the name of output file.')
    print('        <draw> is the flag for drawing the landmarks and showing it. It can be skipped.')
    print('        <save> is the flag for saving resultant image with landmarks drawn on it. It can be skipped.')
    exit()
else:
    video_file = sys.argv[1]
    feature_file_name = "demo.csv"
    draw_flag = False
    save_flag = False
    if len(sys.argv) > 3:
        draw_flag = bool(sys.argv[3])
    if len(sys.argv) > 4:
        save_flag = bool(sys.argv[4])
    print(draw_flag,save_flag)
    df = extractMediaPipeFeatures(video_file,draw_flag,save_flag)
    df.to_csv(feature_file_name,index=False)
    print('Landmark features have been saved in ',feature_file_name)
