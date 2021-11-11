import pandas as pd
import os
from moviepy.editor import *
import datetime
import pandas as pd
import sys


def generateFeatures(video_dir,log_file,vad_file,speech_file,window,output_file_prefix,mapping_file):
    dirlist = dict()
    for f in os.listdir(video_dir):
        f_split = f.split("_")
        if len(f_split) == 6:
            session =  f_split[0]
            group =  f_split[1]
            if group not in dirlist.keys():
                dirlist[group] = list()
            #f_name = video_dir + "/" + f
            f_name = f
            dirlist[group].append(f_name)

    # extract timestamp from video file names


    for key in dirlist.keys():
        print('Group:',key)
        print('============================')
        duration = []
        files = dirlist[key]

        ts = [int(f.split(".")[0].split("_")[5]) for f in files]

        users = [int(f.split(".")[0].split("_")[2]) for f in files]

        duration = [VideoFileClip(video_dir + "/" + vid).duration for vid in files ]

        ts_start = [(t/1000-duration[i]) for i,t in enumerate(ts)]

        ts_end = [t/1000  for t in ts]

        ts_end_diff = [ ( t - min(ts_end))/1000 for t in ts_end]

        ts_start_diff_tmp = [ max(ts_start) - t for t in ts_start]

        ts_start_diff  = [t   for t in ts_start_diff_tmp]


        start_time = datetime.datetime.fromtimestamp(int(max(ts_start)))
        end_time = datetime.datetime.fromtimestamp(int(max(ts_end)))


        output_file_name = output_file_prefix + '_group_' + str(key) + '.csv'

        print('start:',start_time,'end_time:',end_time)
        #dddd = feature_level_fusion('BT19_logs.csv','BT19_vad.csv','BT19_speech.csv',start_time,end_time,window,1,'demo.csv','BT19_mapping.csv')
        dddd = feature_level_fusion(log_file,vad_file,speech_file,start_time,end_time,window,1,output_file_name,mapping_file)

        dddd.to_csv(output_file_name,index=False)


if len(sys.argv) < 7:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python create_split_screen.py <video_directory_path> <log_file> <vad_file> <speech_file> <window:30 seconds> <prefix> <mapping_file> ")
    print('\nhere \n        <video_directory_path> is directory containing merged files which were obtained after merging CoTrack files.')
    print('        <logs_file> is the Etherpad logs file.')
    print('        <vad_file> is the VAD file.')
    print('        <speech_file> is the speech file.')
    print('        <window> is the time window.')
    print('        <prefix> is the prefix for output files.')
    print('        <mapping_file> is the mapping file from CoTrack.')

    exit()
else:
    video_dir = sys.argv[1]
    log_file = sys.argv[2]
    vad_file = sys.argv[3]
    speech_file = sys.argv[4]
    window = pd.Timedelta('30 seconds')
    prefix = sys.argv[6]
    mapping_file = sys.argv[7]
    generateFeatures(video_dir,log_file,vad_file,speech_file,window,prefix,mapping_file)
