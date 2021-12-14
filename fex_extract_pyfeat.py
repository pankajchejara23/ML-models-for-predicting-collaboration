from feat import Detector
import cv2
import argparse
import sys
import os

face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
facepose_model = "img2pose"

detector = Detector(
                        face_model = face_model,
                        landmark_model = landmark_model,
                        au_model = au_model,
                        emotion_model = emotion_model,

                    )
if len(sys.argv) < 3:
    print('Incorrect use, please use the script in the following way.\n')
    print("Use this format: \n python fex_extract_pyfeat.py <source_directory> <target_directory>")
    print('\n      here \n        <source_directory> is the name of the directory having video files')
    print('\n      here \n        <target_directory> is the name of the directory to save feature file')
    exit()
else:
    directory = sys.argv[1]
    target = sys.argv[2]
    print('  Processing started')
    print()
    for f in os.listdir(directory):
        full_file_name = directory + "/" + f
        f_split = f.split(".")[0].split("_")

        if len(f_split) == 6 and f_split[5] == 'VIDEO' and f.split(".")[1] == 'mov' and f_split[2]!= '566':
            print('    Processing file:',f)
            feature_file_ele = f_split[:3] + ['face','pose','hand','pyfeat']
            feature_file_name = '_'.join(feature_file_ele)
            feature_file_name = target + '/' + feature_file_name + '.csv'
            # skip 12 frames
            result = detector.detect_video(full_file_name,skip_frames=12)
            result.to_csv(feature_file_name,index=False)
