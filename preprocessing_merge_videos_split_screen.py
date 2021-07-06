!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:45:36 2021

@author: Pankaj Chejara
"""

from moviepy.editor import *
import os
files = ["2_1_53_Final_file_1625056127928","2_1_54_Final_file_1625056133641","2_1_55_Final_file_1625056120052","2_1_56_Final_file_1625056122009"]

"""
Function to merge multiple videos into single with split screen
Params:
    file_list: list of file_names with timestamp at which file was stored.

"""
def moviepy_create_split_screen(file_list):

    duration = []
    f_clips = []

    # create .mkv file
    for f in files:
        command = "ffmpeg -i "+f+".webm -c copy "+f+".mkv"
        #os.system(command)


    session = file_list[0].split("_")[0]
    group = file_list[0].split("_")[1]

    final_file_name = "Video_" + str(session) + "_" + str(group) + ".mov"

    # extract timestamp and user information
    users = [int(f.split("_")[2])  for f in file_list]

    # timestamp for each video
    ts = [int(f.split("_")[5]) for f in file_list]


    # find duration for each video file
    for f in files:
        file_name = f + ".mkv"
        my_video = VideoFileClip(file_name)
        duration.append(my_video.duration)

    ts_start = [(t/1000-duration[i]) for i,t in enumerate(ts)]

    ts_start_diff_tmp = [ max(ts_start) - t for t in ts_start]

    ts_start_diff  = [ t + .10 if t > 0 else t  for t in ts_start_diff_tmp]

    ts_start_diff = [t + .40 if t == max(ts_start_diff) else t for t in ts_start_diff]

    ts_end = [t/1000 for t in ts]

    ts_end_diff = [ ( t - min(ts_end))/1000 for t in ts_end]


    for i,user in enumerate(users):
        file_name = file_list[i] + ".mkv"
        my_video = VideoFileClip(file_name).subclip(ts_start_diff[i],duration[i] - ts_end_diff[i]).margin(5)
        text = "user_" + str(user)
        my_text = TextClip(text, font ="Arial-Bold", fontsize = 40, color ="red")
        txt_col = my_text.on_color(size=(my_video.w,my_video.h),color=(0,0,0),pos=(15,15),col_opacity=0.1)
        f_clip = CompositeVideoClip([my_video,txt_col])
        f_clips.append(f_clip)

    final_clip = clips_array([[f_clips[0], f_clips[1]],
                          [f_clips[2], f_clips[3]]])

    final_clip.subclip(0,my_video.duration).write_videofile(final_file_name,fps=24,codec='libx264')

moviepy_create_split_screen(files)
