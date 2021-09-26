"""
This script takes media files with time of saving and creates a single synchonised split screen video.

author: Pankaj chejara
"""
from moviepy.editor import *
from os import walk
import functools
import os
import sys

dirlist= []

if len(sys.argv) < 3:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python create_split_screen.py <source_direcotyr> <target_directory>")
    print('\n here <source_directory> is directory containing merged files which were obtained after merging CoTrack files.')
    print('        <target_directory> is the directory where you want to save the split screen files.\n\n')
    exit()
else:
    source_directory = sys.argv[1]
    target_directory = sys.argv[2]


if not os.path.isdir(target_directory):
    os.makedirs(target_directory)

dirlist = {}

for f in os.listdir(source_directory):
    f_split = f.split("_")
    if len(f_split) == 6:
        session =  f_split[0]
        group =  f_split[1]
        if group not in dirlist.keys():
            dirlist[group] = list()
        dirlist[group].append(f)


def moviepy_create_split_screen(files,target_directory=None):
    duration = []
    f_clips = []
    session = files[0].split("_")[0]
    group = files[0].split("_")[1]
    final_file_name = target_directory + "/" + "Video_" + str(session) + "_" + str(group) + ".mov"

    # extract timestamp and user information
    users = [int(f.split("_")[2])  for f in files]

    # timestamp for each video
    ts = [int(f.split("_")[5].split(".")[0]) for f in files]

    # find duration for each video file
    for f in files:

        f_name = f.split(".")[0]
        input_file = source_directory + "/" + f_name
        output_file = target_directory + "/" + f_name
        if not os.path.exists(output_file):
            command = "ffmpeg -i "+ input_file +".webm -c copy "+ output_file +".mkv"
            os.system(command)
        mkv_file_name = output_file + ".mkv"
        my_video = VideoFileClip(mkv_file_name)
        duration.append(my_video.duration)

    ts_start = [(t/1000-duration[i]) for i,t in enumerate(ts)]

    ts_start_diff_tmp = [ max(ts_start) - t for t in ts_start]

    ts_start_diff  = [ t   for t in ts_start_diff_tmp]

    #ts_start_diff = [t + .40 if t == max(ts_start_diff) else t for t in ts_start_diff]

    print([t/60 for t in duration])
    ts_end = [t/1000 + e  for t,e in zip(ts,duration)]
    ts_end_diff = [ ( t - min(ts_end))/1000 for t in ts_end]
    print([t/60 for t in ts_start_diff])
    print(ts_end_diff)

    for i,user in enumerate(users):
        file_name = target_directory +  "/" + files[i].split(".")[0] + ".mkv"
        my_video = VideoFileClip(file_name).subclip(ts_start_diff[i],duration[i]).margin(5)
        wr_file_name = target_directory + "/" + files[i].split(".")[0] + '_IND' + '.mkv'
        wr_audio_file_name = target_directory + "/" + files[i].split(".")[0] + '_audio_IND' + '.wav'
        #my_video.audio.write_audiofile(wr_audio_file_name,ffmpeg_params=["-ac", "1"],fps=32000,nbytes=2)
        #my_video.write_videofile(wr_file_name,fps=24,codec='libx264')
        text = "user_" + str(user)
        my_text = TextClip(text, font ="Arial-Bold", fontsize = 40, color ="red")
        txt_col = my_text.on_color(size=(my_video.w,my_video.h),color=(0,0,0),pos=(15,15),col_opacity=0.1)
        f_clip = CompositeVideoClip([my_video,txt_col])
        f_clips.append(f_clip)

    if len(f_clips) == 2:
        final_clip = clips_array([[f_clips[0], f_clips[1]]])

    if len(f_clips) == 3:
        height = f_clips[0].size[1] + f_clips[1].size[1]
        width = f_clips[0].size[0] + f_clips[1].size[0]
        final_clip = CompositeVideoClip([f_clips[0].set_position((325,0)),f_clips[1].set_position((0,490)),f_clips[2].set_position((625,490))],size=(width,height))
    if len(f_clips) == 4:
        final_clip = clips_array([[f_clips[0], f_clips[1]],[f_clips[2],f_clips[3]]])


    final_clip.subclip(0,my_video.duration).write_videofile(final_file_name,fps=24,codec='libx264')


for key,item in dirlist.items():
    print('\nGroup:',key)
    print("------------------------")
    print('Processing...')
    moviepy_create_split_screen(dirlist[key],target_directory)
    print('Split screen video file is created and saved in ',target_directory)
    #print("\n".join([f for f in dirlist[key]]))
    print("------------------------")
