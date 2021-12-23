import pandas as pd
from scipy import stats
import numpy as np
import os
import math
import sys
import cv2
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def SpEn(sig,ordr,tor):
    # https://github.com/yzjba/sampen/blob/master/sampen.py
    # sig: the input signal or series, it should be numpy array with type float
    # ordr: order, the length of template
    # tor: percent of standard deviation

    n = len(sig)
    tor = np.std(sig)*tor

    matchnum = 0.0
    for i in range(n-ordr):
        tmpl = sig[i:i+ordr]
        for j in range (i+1,n-ordr+1):
            ltmp = sig[j:j+ordr]
            diff = tmpl-ltmp
            if (diff<tor).all():
                matchnum+=1

    allnum = (n-ordr+1)*(n-ordr)/2
    if matchnum<0.1:
        sen = 1000.0
    else:
        sen = -math.log(matchnum/allnum)
    return sen

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

def getMouthArea(shape):
    mouth_landmarks = list(range(48,60)) + [48]
    coords = []
    for p in mouth_landmarks:
        coords.append(getPoint(shape,p))
    area = cv2.contourArea(np.array(coords).astype(np.float32))
    return area

def getPoint(ex,n):
    x_label = 'x_' + str(n)
    y_label = 'y_' + str(n)
    x = int(ex[x_label])
    y = int(ex[y_label])
    return (x,y)

def addHeadPoseMouthArea(rows,shape):
    head_rotate_x = []
    head_rotate_y = []
    head_rotate_z = []

    face_area = []

    mouth_area = []

    for i in range(rows.shape[0]):
        row = rows.iloc[i,:]

        if row['FaceScore'] > .50:
            head = estimateHeadPose(row,shape)
            mth = getMouthArea(row)
            head_rotate_x.append(head[0])
            head_rotate_y.append(head[1])
            head_rotate_z.append(head[2])
            face_area.append((row['FaceRectHeight']*row['FaceRectWidth']))

            mouth_area.append(mth)

    ndf = pd.DataFrame({'head_rotate_x':head_rotate_x,'head_rotate_y':head_rotate_y,'head_rotate_z':head_rotate_z,'mouth_area':mouth_area,'face_area':face_area})

    result = pd.concat([rows,ndf])

    return result


def aggregatePerSecond(f1,img_shape=(490,650,3)):
    second_level_df_cols = ['second'] + col_names
    df_level1 = pd.DataFrame(columns = second_level_df_cols)

    window = 1
    final_df = []
    min_instances = df.shape[0]
    fps = 24
    no_skip_frames = 6
    data_fps = (fps/no_skip_frames)
    for i in range(int(min_instances/data_fps)):
        rows = []
        start_frame = i * 24
        end_frame = (i+1) * 24
        rows = f1.loc[(f1['frame'] >= start_frame) & (f1['frame'] < end_frame),:]
        rows = addHeadPoseMouthArea(rows,img_shape)
        ag_features = rows.loc[:,col_names]
        per_second = ag_features.mean(axis=0).to_dict()

        per_second['second'] = i+1
        df_level1 = df_level1.append(per_second,ignore_index=True)
    return df_level1

def aggregatePerWindow(df_level1):
    no_seconds = df_level1.shape[0]
    # spw: seconds per window
    spw = 30
    window_level_df_cols = ['window']

    for stat in stats:
        for col in col_names:
            column = col + '_' + stat
            window_level_df_cols.append(column)
    level2_df = pd.DataFrame(columns = window_level_df_cols)

    print(window_level_df_cols)

    for i in range(int(no_seconds/spw)):
        rows = []
        start_frame = i * 24
        end_frame = (i+1) * 24
        rows = df_level1.loc[(df_level1['second'] >= start_frame) & (df_level1['second'] < end_frame),col_names]
        per_window = {}
        for ind,stat in enumerate(stats):
            for col in col_names:
                column = col + '_' + stat
                data = rows[col].to_numpy()
                if stat in ['sample_entropy']:
                    per_window[column] = np_funcs[ind](data,2,.2)
                else:
                    per_window[column] = np_funcs[ind](data)
        per_window['window'] = i+1
        level2_df = level2_df.append(per_window,ignore_index=True)
    return level2_df


col_names = ['face_area','mouth_area','head_rotate_x','head_rotate_y','head_rotate_z','mouth_area','AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU11','AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43','anger','disgust','fear','happiness','sadness','surprise','neutral',]
stats = ['mean','median','min','max','std','skew','kurtosis','range','sample_entropy']
np_funcs = [np.mean,np.median,np.amin,np.amax,np.std,np.ptp,skew,kurtosis,SpEn]


if len(sys.argv) < 4:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python fexFeatureAgg.py <video_directory> <source_direcotyr> <target_directory>")
    print('\n here <video_directory> is directory containing video files.')
    print('\n      <source_directory> is directory containing py-feat files.')
    print('        <target_directory> is the directory where you want to save group-level merged features.\n\n')
    exit()
else:
    video_directory = sys.argv[1]
    source_directory = sys.argv[2]
    target_directory = sys.argv[3]


if not os.path.isdir(target_directory):
    os.makedirs(target_directory)

dirlist = {}

for f in os.listdir(source_directory):
    f_split = f.split("_")
    if len(f_split) == 9:
        session =  f_split[0]
        group =  f_split[1]
        if group not in dirlist.keys():
            dirlist[group] = list()
        dirlist[group].append(f)


#print(dirlist)
# generate window wise features
for group in dirlist.keys():
    files = dirlist[group]
    group_df = []
    for file in files:
        print('[Window features]: Processing :',file)
        file_ele = file.split('.')[0].split('_')
        file = source_directory + '/' + file
        video_file_name = video_directory + "/" + "_".join(file_ele[:3]) +"_SYNC_IND_VIDEO.mov"
        if os.path.isfile(video_file_name):
            cap = cv2.VideoCapture(video_file_name)
            while cap.isOpened():
                success,img = cap.read()
                break
            cap.release()
        else:
            print('Video file not found')
            break
        df = pd.read_csv(file)
        try:
            df1 = aggregatePerSecond(df,img.shape)
        except:
            df1 = aggregatePerSecond(df)
        df2 = aggregatePerWindow(df1)
        group_df.append(df2)
        result_file = "_".join(file_ele[:4]+['fex','30s_window']) + '.csv'
        result_file = target_directory + "/" + result_file
        df2.to_csv(result_file,index=False)
