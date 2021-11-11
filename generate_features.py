import pandas as pd
import os
from moviepy.editor import *
import datetime
import pandas as pd
import sys

def feature_level_fusion(log_file,vad_file,speech_file,start_time,end_time,window,group,output_file_name,mapping_file):
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
    overlaps = {}

    # opening file
    log = pd.read_csv(log_file)
    vad = pd.read_csv(vad_file)
    speech = pd.read_csv(speech_file)

    # user mapping file
    mapping = pd.read_csv(mapping_file)
    user_mapping = dict()
    for i in range(mapping.shape[0]):
        record = mapping.iloc[i,:]
        user_mapping[record['authorid']] = record['username']


    #changing datatype of timestamp column
    log.timestamp = pd.to_datetime(log.timestamp,format="%H:%M:%S %d-%m-%Y")
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
    authors = set(log.loc[log['group'] == int(group),:]['author'].values )

    for author in authors:
        added[author] = []
        deleted[author] = []
        speak[author] = []
        turns[author] = []
        speech_text[author] = []


    while(frame_start <= end_time):
        print('========================')

        frame_end = frame_start + window
        #print('frame start:',frame_start,'frame_end:',frame_end)
        print('Frame:',frame_no)

        # fetching data records for the specified window
        mask_log = (log['timestamp'] > frame_start) & (log['timestamp'] <= frame_end)
        mask_vad = (vad['timestamp'] > frame_start) & (vad['timestamp'] <= frame_end)
        mask_speech = (speech['timestamp'] > frame_start) & (speech['timestamp'] <= frame_end)

        df = log[mask_log]
        df_vad = vad[mask_vad]
        df_speech = speech[mask_speech]


        sequence = df_speech['user'].to_list()

        # For computing turn-taking
        turn_df = pd.DataFrame(columns=['label','conti_frequency'])

        # This function will count the number of continuous occurence
        def count_conti_occurence(index):

            # Set count to 0
            count=0

            # Starts from the given index
            j = index

            # Loop to iterate over the users sequence
            while j<len(sequence):

                # Increase the count if the element at given index (parameter) is same as the iterated element
                if sequence[j] == sequence[index]:
                    count +=1

                # If mismatch found, break the loop
                else:
                    break

                # Increases j
                j +=1

            # Return number of count for sequence[index] and index of first next occurence of different element.
            return count,(j-index)

        # Set i to 0 for the Loop
        i = 0

        # Iterate for entire sequence of users
        while i < len(sequence):

            # Call count_conti_occurence() function
            count,diff = count_conti_occurence(i)


            # Add continuous frequency of current user (sequence[i]) to the dataframe
            turn_df = turn_df.append({'label':sequence[i],'conti_frequency':count},ignore_index=True)


            # Move to next different element
            i = i + diff

        if df.shape[0] != 0:
            # Etherpad text
            text.append(df["text"].values[-1])
        else:
            text.append("")

        for author in authors:

            author_df = df.loc[df['author'] == author,['operation','difference']]
            author_vad_df = df_vad.loc[df_vad['user'] == author,:]
            author_speech_df = df_speech.loc[df_speech['user'] == author,:]



            if author_df.shape[0] != 0:
                author_add = author_df[author_df['operation'] == '>']['difference'].sum()
                author_del = author_df[author_df['operation'] == '<']['difference'].sum()
                added[author].append(author_add)
                deleted[author].append(author_del)
            else:
                added[author].append(0)
                deleted[author].append(0)

            if author_vad_df.shape[0] != 0:
                #print(author_vad_df['speaking_time(sec.)'].sum())
                speak[author].append(author_vad_df['speaking_time(sec.)'].sum())
            else:
                speak[author].append(0)

            if author_speech_df.shape[0] != 0:
                speech_text[author].append(author_speech_df['speech'].values)
            else:
                speech_text[author].append('')


            turns[author].append(turn_df.loc[turn_df['label']==author,:].shape[0])

        frame_start = frame_end
        frames.append(frame_no)
        frame_no = frame_no +  1

    users = {}
    users['frame'] = frames

    for i,author in enumerate(authors):
        user_add = 'user' + str(user_mapping[author]) +'_add'
        user_del = 'user' + str(user_mapping[author]) +'_del'
        user_speak = 'user' + str(user_mapping[author]) +'_speak'
        user_speech = 'user' + str(user_mapping[author]) +'_speech'
        user_turns ='user' + str(user_mapping[author]) +'_turns'
        users[user_add] = added[author]
        users[user_del] = deleted[author]
        users[user_speak] = speak[author]
        users[user_speech] = speech_text[author]
        users[user_turns] = turns[author]
    users['text'] = text
    group_df = pd.DataFrame(users)
    return group_df
    
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
        print('Generated features for group-',group,' save in file:',output_file_name)


if len(sys.argv) < 7:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python create_split_screen.py <video_directory_path> <log_file> <vad_file> <speech_file> <window:30 seconds> <prefix> <mapping_file> ")
    print('\n      here \n        <video_directory_path> is directory containing merged files which were obtained after merging CoTrack files.')
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
