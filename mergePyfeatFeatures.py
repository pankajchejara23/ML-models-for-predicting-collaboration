import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s','--session',required=True,help="specify the session id")
parser.add_argument('-s','--source_directory',required=True,help="path of directory containing features extracted using pyfeat")
parser.add_argument('-t','--target_directory',required=True,help="path of directory to save the merged features")

args = vars(parser.parse_args())


dirlist = {}
session = args['session']
source_directory = args['source_directory']

for f in os.listdir(source_directory):
    f_split = f.split(".")[0].split("_")
    if (len(f_split) == 7) and (f_split[6] == 'pyfeat'):
        session =  f_split[0]
        group =  f_split[1]
        if group not in dirlist.keys():
            dirlist[group] = list()
        dirlist[group].append(f)


for group in dirlist.keys():
    print('Processing group:',group)
    file_name_ele = [session,group,'group','face','hand','pyfeat']
    df_list = []
    for fl in dirlist[group]:
        full_file_name = source_directory + "/" + fl
        df = pd.read_csv(full_file_name)
        df_list.append(df)
    window = 1
    final_df = []
    min_instances = min([df.shape[0] for df in df_list])
    no_skip_frames = 6
    for i in range(int(min_instances/no_skip_frames)):
        rows = []
        for g in groups:
            row = g.loc[(g['frame'] <= window * 12) & (g['frame'] > (window-1)*12),:]
            rows.append(row)
        window_df = pd.concat(rows)
        window_df_mean = window_df.mean(axis=0).to_frame().T
        final_df.append(window_df_mean)
        window += 1
    file_name = target_directory + "/" + '_'.join(file_name_ele) + '.csv'
    merge_df = pd.concat(final_df)
    merge_df.to_csv(file_name,index=False)
    print(' Merged features for group %s stored in file %s'%(group,file_name))
    print(merge_df.shape)
