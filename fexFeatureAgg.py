import pandas as pd
from scipy import stats
import numpy as np
import os
import math
import sys


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
            if all(diff<tor):
                matchnum+=1

    allnum = (n-ordr+1)*(n-ordr)/2
    if matchnum<0.1:
        sen = 1000.0
    else:
        sen = -math.log(matchnum/allnum)
    return sen


def aggregatePerSecond(f1):
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
        rows = f1.loc[(f1['frame'] >= start_frame) & (f1['frame'] < end_frame),col_names]
        per_second = rows.mean(axis=0).to_dict()

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


col_names = ['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU11','AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43','anger','disgust','fear','happiness','sadness','surprise','neutral',]
stats = ['mean','median','min','max','std','range','sample_entropy']
np_funcs = [np.mean,np.median,np.amin,np.amax,np.std,np.ptp,SpEn]


if len(sys.argv) < 3:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python fexFeatureAgg.py <source_direcotyr> <target_directory>")
    print('\n here <source_directory> is directory containing py-feat files.')
    print('        <target_directory> is the directory where you want to save group-level merged features.\n\n')
    exit()
else:
    source_directory = sys.argv[1]
    target_directory = sys.argv[2]


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
    for file in files:
        print('[Window features]: Processing :',file)
        file_ele = file.split('.')[0].split('_')
        file = source_directory + '/' + file
        df = pd.read_csv(file)
        df1 = aggregatePerSecond(df)
        df2 = aggregatePerWindow(df1)
        result_file = "_".join(file_ele[:4]+['fex','30s_window']) + '.csv'
        result_file = target_directory + "/" + result_file
        df2.to_csv(result_file,index=False)
