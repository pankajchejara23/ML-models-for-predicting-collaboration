def moviepy_create_split_screen(files):
    """
    This script is for the cases where single user has more than one final files.
    In such cases, the processed file has extension 'mov' and that's why this script doesn't rely on 'mkv' file type.
    """
    duration = []
    f_clips = []
    session = files[0].split("_")[0]
    group = files[0].split("_")[1]
    final_file_name =  "Video_" + str(session) + "_" + str(group) + ".mov"

    # extract timestamp and user information
    users = [int(f.split("_")[2])  for f in files]

    # timestamp for each video
    ts = [int(f.split("_")[5].split(".")[0]) for f in files]

    # find duration for each video file
    for f in files:
        my_video = VideoFileClip(f)
        duration.append(my_video.duration)

    ts_start = [(t/1000-duration[i]) for i,t in enumerate(ts)]
    ts_end = [t/1000 + e  for t,e in zip(ts,duration)]
    ts_end_diff = [ ( t - min(ts_end))/1000 for t in ts_end]

    ts_start_diff_tmp = [ max(ts_start) - t for t in ts_start]

    ts_start_diff  = [ t   for t in ts_start_diff_tmp]
    #print([t/60 for t in ts_start_diff])
    print(ts_end_diff)

    for i,user in enumerate(users):
        file_name = files[i]
        #print("file details: ",file_name,ts_start_diff[i],duration[i])
        my_video = VideoFileClip(file_name).subclip(ts_start_diff[i],duration[i]).margin(5)
        if my_video.w > 1000:
            my_video = my_video.resize((640,480))
        else:
            my_video = my_video

        # synchronized timestamp for current video file
        cur_file_start_ts = int(files[i].split("_")[5].split(".")[0]) - duration[i] + ts_start_diff[i]
        output_file_name = str(session) + '_' + str(group) + '_' + str(user) + '_SYNC_IND_' + str(int(cur_file_start_ts))



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
