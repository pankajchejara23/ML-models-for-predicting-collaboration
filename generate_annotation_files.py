import pandas as pd
import os
import datetime

window = pd.Timedelta('30 seconds')


def generate_annotation_file_for_group(user_mapping_file,log_file,start_time,end_time,window,group,output_file_name):
    """
    user_mapping_file: mapping file from CoTrack
    start_time: start time taken from sync videos of all participants
    end_time: max end time from the participants
    window: size of window
    group: group number
    output_file_name: Name of output file

    Feature generated
    Etherpad logs feature:
    #1 : number of chars added (user-level)
    #2 : number of chars deleted (user-level)

    """

    # initializing
    frames = []         # storing frame numbers
    added = {}          # storing number of chars added by each user
    deleted = {}        # storing number of chars deleted by each user
    frame_no = 1
    # opening file
    log = pd.read_csv(log_file)
    #changing datatype of timestamp column
    log.timestamp = pd.to_datetime(log.timestamp)
    # localize time in vad and speech data frame
    # CoTrack server adds timezone info into the timestmap while etherpad sever provides without timezone.
    print('Group:',group)

    frame_start = start_time
    # fetching all authors from specified group

    authors = set(log.loc[log['group']==int(group),:]['author'].values)

    for author in authors:
        added[author] = []
        deleted[author] = []


    while(frame_start <= end_time):
        #print('frame start:',frame_start)
        frame_end = frame_start + window

        # fetching data records for the specified window
        mask_log = (log['timestamp'] > frame_start) & (log['timestamp'] <= frame_end)

        df = log[mask_log]



        for author in authors:
            author_df = df.loc[df['author'] == author,['operation','difference']]

            if author_df.shape[0] != 0:
                author_add = author_df[author_df['operation'] == '>']['difference'].sum()
                author_del = author_df[author_df['operation'] == '<']['difference'].sum()
                added[author].append(author_add)
                deleted[author].append(author_del)
            else:
                added[author].append(0)
                deleted[author].append(0)

        frame_start = frame_end
        frames.append(frame_no)
        frame_no = frame_no +  1
    users = {}
    users['frame'] = frames

    mapping = pd.read_csv(user_mapping_file)
    user_mapping = dict()
    for i in range(mapping.shape[0]):
        record = mapping.iloc[i,:]
        user_mapping[record['authorid']] = record['username']

    users['SMU'] = [' '] * len(frames)
    users['ARG'] = [' '] * len(frames)
    users['KE'] = [' '] * len(frames)
    users['CF'] = [' '] * len(frames)
    users['CO'] = [' '] * len(frames)
    users['STR'] = [' '] * len(frames)
    users['ITO'] = [' '] * len(frames)




    for i,author in enumerate(authors):
        user_add = 'user-' + str(user_mapping[author]) +'_add'
        user_del = 'user-' + str(user_mapping[author]) +'_del'
        users[user_add] = added[author]
        users[user_del] = deleted[author]
        print('Author:',author,'length:',len(added[author]))
    group_df = pd.DataFrame(users)
    group_df.to_csv(output_file_name,index=False)
    return group_df


def generate_annotation_files(video_dir,output_file_prefix,mapping_file,log_file):
    """
    video_dir: directory containing video files,
    output_file_prefix: prefix to save output file,
    mapping_file: mapping file from CoTrack
    log_file: log file from CoTrack
    """
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
        output_file = output_file_prefix +'group_'+str(key)+'.csv'
        df = generate_annotation_file_for_group(mapping_file,log_file,start_time,end_time,window,key,output_file)

        #df = generate_annotation_file_for_group('/Users/pankaj/Documents/CoTrack2_datasets/session_39_mkv','ITA20 Sept 30th_mapping.csv','ITA20 Sept 30th_logs.csv',start_time,end_time,window,key,output_file)

video_dir = '/Users/pankaj/Documents/CoTrack2_datasets/session_39_mkv'
generate_annotation_files(video_dir,'session_39_','ITA20 Sept 30th_mapping.csv','ITA20 Sept 30th_logs.csv')
