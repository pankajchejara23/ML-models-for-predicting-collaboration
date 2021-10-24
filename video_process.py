"""
This script takes media files with time of saving and creates a single synchonised split screen video.

author: Pankaj chejara
"""
from moviepy.editor import *
from os import walk
import functools
import os
import sys
import functools
dirlist= []

exclude_user = None

if len(sys.argv) < 4:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python process_video_files.py <flag> <source_direcotyr> <target_directory>")
    print('\n here flags are gen_mkv: for generating only mkv files of webm, \n    con_mkv: scan for multiple mkv files of single user and concatenate them \n gen_split: create split screen videos')
    print('        <source_directory> is directory containing merged files which were obtained after merging CoTrack files.')
    print('        <target_directory> is the directory where you want to save the split screen files.\n\n')
    exit()
else:
    flag = sys.argv[1]
    source_directory = sys.argv[2]
    target_directory = sys.argv[3]


print(flag)

if not os.path.isdir(target_directory):
    os.makedirs(target_directory)

dirlist = {}
userfilelist = {}

for f in os.listdir(source_directory):
    f_split = f.split("_")
    if len(f_split) == 6:
        session =  f_split[0]
        group =  f_split[1]
        user = f_split[2]
        if group not in dirlist.keys():
            dirlist[group] = list()
        if user not in userfilelist.keys():
            userfilelist[user] = list()
        dirlist[group].append(f)
        if f.split(".")[1] == 'mkv':
            userfilelist[user].append(f)

def compare(item1,item2):
    item1 = item1.split('.')[0]
    item2 = item2.split('.')[0]
    i1 = int(item1.split('_')[5])
    i2 = int(item2.split('_')[5])

    if i1 > i2:
        return 1
    else:
        return -1


def generate_webm_to_mkv(files,source_directory=None,target_directory=None):
    for f in files:
        f_name = f.split(".")[0]
        input_file = source_directory + "/" + f_name
        output_file = target_directory + "/" + f_name

        output_file_name = output_file + ".mkv"

        if not os.path.exists(output_file_name):
            command = "ffmpeg -i "+ input_file +".webm -c copy "+ output_file +".mkv"
            os.system(command)
    print('Command has been executed successfully')


def concatenate_mkv(userfiles,source_directory=None,target_directory=None):
    for key in userfiles.keys():
        if len(userfiles[key]) > 1:

            userfiles[key].sort(key=functools.cmp_to_key(compare))
            print(userfiles[key])
            filelist = [ source_directory + "/" + f for f in userfiles[key]]

            file_list = "|".join(filelist)
            session = userfiles[key][-1].split("_")[0]
            group = userfiles[key][-1].split("_")[1]
            user = userfiles[key][-1].split("_")[2]
            ts = userfiles[key][-1].split("_")[5].split(".")[0]

            output_file_name = str(session) + '_' + str(group) + '_' + str(user) + '_' + str(ts) + '_pr.mkv'
            command = "ffmpeg -i \"concat:"+ file_list +"\" -c:v copy -c:a aac "+ target_directory+"/"+output_file_name
            print('Running: ',command)
            os.system(command)



def moviepy_create_split_screen(files,target_directory=None,exclude_user=None):
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

        output_file_name = output_file + ".mkv"
        if not os.path.exists(output_file_name):
            command = "ffmpeg -i "+ input_file +".webm -c copy "+ output_file +".mkv"
            os.system(command)
            print('Generating mkv')
        mkv_file_name = output_file + ".mkv"
        my_video = VideoFileClip(mkv_file_name)
        duration.append(my_video.duration)

    ts_start = [(t/1000-duration[i]) for i,t in enumerate(ts)]
    ts_end = [t/1000 + e  for t,e in zip(ts,duration)]
    ts_end_diff = [ ( t - min(ts_end))/1000 for t in ts_end]

    ts_start_diff_tmp = [ max(ts_start) - t for t in ts_start]

    ts_start_diff  = [ t   for t in ts_start_diff_tmp]
    #print([t/60 for t in ts_start_diff])
    print(ts_end_diff)

    for i,user in enumerate(users):
        file_name = "./" + target_directory +  "/" + files[i].split(".")[0] + ".mkv"
        #print("file details: ",file_name,ts_start_diff[i],duration[i])
        my_video = VideoFileClip(file_name).subclip(ts_start_diff[i],duration[i]).margin(5)
        if my_video.w > 1000:
            my_video = my_video.resize((640,480))
        else:
            my_video = my_video
        # synchronized timestamp for current video file
        cur_file_start_ts = int(files[i].split("_")[5].split(".")[0]) - duration[i] + ts_start_diff[i]
        output_file_name = str(session) + '_' + str(group) + '_' + str(user) + '_SYNC_IND_'

        wr_file_name = target_directory + "/" + output_file_name + 'VIDEO.mkv'
        wr_audio_file_name = target_directory + "/" + output_file_name + 'AUDIO.wav'
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


    #inal_clip.subclip(0,my_video.duration).write_videofile(final_file_name,fps=24,codec='libx264')

if flag=="gen_mkv":
    print('Calling generating mkv module')
    print(dirlist)
    for key,item in dirlist.items():
        generate_webm_to_mkv(dirlist[key],source_directory,target_directory)
elif flag=="con_mkv":

    concatenate_mkv(userfilelist,source_directory,target_directory)
elif flag=="gen_split":

    for key,item in dirlist.items():
        print('\nGroup:',key)
        print("------------------------")
        print('Processing...')
        moviepy_create_split_screen(dirlist[key],target_directory,exclude_user)
        print('Split screen video file is created and saved in ',target_directory)
        #print("\n".join([f for f in dirlist[key]]))
        print("------------------------")
