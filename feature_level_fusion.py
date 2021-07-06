import pandas as pd

def get_start_end_time(files):
  """
  This script takes filenames of recorded videofiles from CoTrack tool and return aligned start time and end time of the video files.
  As every recorded video can have a different time, therefore, we need to compute start time and end time in a way that each video file has data for that duration.
  """
    users = [int(f.split("_")[2])  for f in files]
    ts = [int(f.split("_")[5]) for f in files]
    duration = []
    for f in files:
        file_name = f + ".mkv"
        my_video = VideoFileClip(file_name)
        duration.append(my_video.duration)
    ts_new = [(t - min(ts))/1000 for t in ts]
    start_time, sdx = min((val,ids) for (ids,val) in enumerate(ts_new))
    st_time =  datetime.datetime.fromtimestamp(int(ts[sdx]/1000 - duration[sdx]))
    print('Start time:',st_time)
    updated_duration = [(d - ts_new[i]) for i,d in enumerate(duration)]
    #print(updated_duration)
    min_duration, idx = min((val,ids) for (ids,val) in enumerate(updated_duration))
    en_time = datetime.datetime.fromtimestamp(int(ts[idx]/1000))
    print('End time:',en_time)
    return st_time,en_time

def feature_level_fusion(log_file,vad_file,speech_file,start_time,end_time,window,group):
    """
    This script takes three files etherpad logs, vad, speech from CoTrack tool and
    perform feature extraction and feature level fusion.

    Etherpad logs feature:
    #1 : number of chars added (user-level)
    #2 : number of chars deleted (user-level)
    #3 : number of words added (user-level)
    #4 : number of words deleted (user-level)
    #5 : number of lines added (user-level)
    #6 : number of users writing in the same window
    #7 : number of users deleting in the same window

    Vad features
    #1 : speaking time (user-level)
    #2 : number of speaking turns (user-level)
    #3 : average speaking time (user-level)
    #4 : silence time
    #5 : simultaneous speaking
    #6 : number of instances when more than one user was seapking

    Speech features
    #1 : number of words (user-level)
    #2 : number of wh-words (user-level)
    #3 : cohesion level (user-level) @todo

    """
    # initializing
    frames = []         # storing frame numbers
    added = {}          # storing number of chars added by each user
    deleted = {}        # storing number of chars deleted by each user
    speak = {}          # storing speaking time of each user (seconds)
    turns = {}          # storing number of speaking turns of each user
    speech_text = {}    # storing speech of each user
    text = []           # storing etherpad text
    frame_no = 1

    # opening file
    log = pd.read_csv(log_file)
    vad = pd.read_csv(vad_file)
    speech = pd.read_csv(speech_file)


    #changing datatype of timestamp column
    log.timestamp = pd.to_datetime(log.timestamp)
    vad.timestamp = pd.to_datetime(vad.timestamp)
    speech.timestamp = pd.to_datetime(speech.timestamp)

    # localize time in vad and speech data frame
    # CoTrack server adds timezone info into the timestmap while etherpad sever provides without timezone.
    vad['timestamp'] = vad['timestamp'].dt.tz_convert('Europe/Helsinki')
    speech['timestamp'] = speech['timestamp'].dt.tz_convert('Europe/Helsinki')

    vad['timestamp'] = vad['timestamp'].dt.tz_localize(None)
    speech['timestamp'] = speech['timestamp'].dt.tz_localize(None)

    frame_start = start_time

    # fetching all authors from specified group
    authors = set(log.loc[log['group'] == group,:]['author'].values)

    for author in authors:
        added[author] = []
        deleted[author] = []
        speak[author] = []
        turns[author] = []
        speech_text[author] = []

    while(frame_start <= end_time):
        #print('frame start:',frame_start)
        frame_end = frame_start + window

        # fetching data records for the specified window
        mask_log = (log['timestamp'] > frame_start) & (log['timestamp'] <= frame_end)
        mask_vad = (vad['timestamp'] > frame_start) & (vad['timestamp'] <= frame_end)
        mask_speech = (speech['timestamp'] > frame_start) & (speech['timestamp'] <= frame_end)

        df = log[mask]
        df_vad = vad[mask_vad]
        df_speech = speech[mask_speech]

        if df.shape[0] != 0:
            # Etherpad text
            text.append(df["text"].values[-1])
        else:
            text.append("")

        for author in authors:
            author_df = df.loc[df['author'] == author,['operation','difference']]
            author_vad_df = df_vad.loc[df_vad['user'] == author,:]
            author_speech_df = df_speech.loc[df_speech['user'] == author]

            if author_df.shape[0] != 0:
                author_add = author_df[author_df['operation'] == '>']['difference'].sum()
                author_del = author_df[author_df['operation'] == '<']['difference'].sum()
                added[author].append(author_add)
                deleted[author].append(author_del)
            else:
                added[author].append(0)
                deleted[author].append(0)

            if author_vad_df.shape[0] != 0:
                print(author_vad_df['speaking_time(sec.)'].sum())
                speak[author].append(author_vad_df['speaking_time(sec.)'].sum())
            else:
                speak[author].append(0)

            if author_speech_df.shape[0] != 0:
                speech_text[author].append(author_speech_df['speech'].values)
            else:
                speech_text[author].append('')


        frame_start = frame_end
        frames.append(frame_no)
        frame_no = frame_no +  1
    users = {}
    users['frame'] = frames

    for i,author in enumerate(authors):
        user_add = 'user' + str(i+1)+'_add'
        user_del = 'user' + str(i+1)+'_del'
        user_speak = 'user' + str(i+1)+'_speak'
        user_speech = 'user' + str(i+1)+'_speech'
        users[user_add] = added[author]
        users[user_del] = deleted[author]
        users[user_speak] = speak[author]
        users[user_speech] = speech_text[author]
    users['text'] = text
    group_df = pd.DataFrame(users)
    return group_df
