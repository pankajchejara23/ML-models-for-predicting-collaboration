import cv2
import sys
import mediapipe as mp
import math
import pandas as pd
from typing import List, Mapping, Optional, Tuple, Union

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# this threshold will determine whether we the detected landmark is considered present or not.
# idea is taken from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


# this function is also taken from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
# As MediaPipe return landmarks in normalized form, we need to convert them back relative to the image.
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def extractMediaPipeFeatures(video_file,draw=False,save=False):
    cap = cv2.VideoCapture(video_file)
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
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



            results = holistic.process(image)

            # we need image height and width for converting normalized coordinates back to original
            image_rows, image_cols, _ = image.shape

            # Draw landmark annotation on the image.
            image.flags.writeable = True

            idx_to_coordinates = {}

            for idx, landmark in enumerate(results.face_landmarks.landmark):
                x_label = str(idx) + '_x'
                y_label = str(idx) + '_y'

                if ((landmark.HasField('visibility') and
                     landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < _PRESENCE_THRESHOLD)):
                    continue
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
                if landmark_px:
                    idx_to_coordinates[x_label] = landmark_px[0]
                    idx_to_coordinates[y_label] = landmark_px[1]
                else:
                    # if landmark is not present than insert empty values
                    idx_to_coordinates[x_label] = ''
                    idx_to_coordinates[y_label] = ''

            # append the landmark to the dataframe
            output_df = output_df.append(idx_to_coordinates,ignore_index=True)

            if draw:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                cv2.imshow('MediaPipe Pose', image)
            if save:
                fname = 'img_' + str(i) + '.jpg'
                cv2.imwrite(fname,image)

            print('Processing frame:',i)
            i = i + 1

    cap.release()
    return output_df


if len(sys.argv) < 3:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python extractFaceLandmarks.py <video_file> <feature_file_name> <draw> <save>")
    print('\n      here \n        <video_file> name of the video file with the path')
    print('        <feature_file_name> is the name of output file.')
    print('        <draw> is the flag for drawing the landmarks and showing it. It can be skipped.')
    print('        <save> is the flag for saving resultant image with landmarks drawn on it. It can be skipped.')


    exit()
else:
    video_file = sys.argv[1]
    feature_file_name = sys.argv[2]
    draw_flag = False
    save_flag = False
    if len(sys.argv) > 3:
        draw_flag = bool(sys.argv[3])
    if len(sys.argv) > 4:
        save_flag = bool(sys.argv[4])
    print(draw_flag,save_flag)
    df = extractMediaPipeFeatures(video_file,draw_flag,save_flag)
    df.to_csv(feature_file_name,index=False)
    print('Landmark features have been save in ',feature_file_name)
