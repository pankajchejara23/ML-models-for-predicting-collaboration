import pandas as pd
from scipy.stats import entropy
import math
import numpy as np
import argparse


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def mergeFeatures(files,groups):
    """
    This function takes features files generate by CoTrack and merge them for machine learning analysis.
    This script must be run on the files generate by generate_features.py scripts.
    # params
    files: list of file names having features (CoTrack featuers)
    groups: a list of group names in the same sequence of file names.

    """
    features = {}
    columns = ['frame','group','user_add_mean','user_add_sd','user_del_mean','user_del_sd',
           'user_speak_mean','user_speak_sd','user_turns_mean','user_turns_sd',
           'user_add_entropy','user_del_entropy','user_speak_entropy','user_turns_entropy',
           'user_speech','user_add_gini','user_del_gini','user_speak_gini',
           'user_turns_gini',]

    for index,file in enumerate(files):
        group = groups[index]
        fet1 = pd.read_csv(file)
        print(fet1.shape)
        fet1['group'] = group
        speech_cols = []
        speak_cols = []
        turns_cols = []
        add_cols = []
        del_cols = []
        tmp_df = pd.DataFrame(columns = columns)
        all_cols = {'speech':speech_cols,'speak':speak_cols,'turns':turns_cols,'add':add_cols,'del':del_cols}

        for col in fet1.columns:
            sp = col.split('_')
            if len(sp) == 2 and sp[1] == 'speech':
                speech_cols.append(col)
            elif len(sp) == 2 and sp[1] == 'speak':
                speak_cols.append(col)
            elif len(sp) == 2 and sp[1] == 'turns':
                turns_cols.append(col)
            elif len(sp) == 2 and sp[1] == 'add':
                add_cols.append(col)
            elif len(sp) == 2 and sp[1] == 'del':
                del_cols.append(col)

        for ind in range(fet1.shape[0]):
            tmp_df_row = {}

            tmp_df_row['frame'] = fet1.loc[ind,'frame']
            tmp_df_row['group'] = fet1.loc[ind,'group']

            for type_cols  in all_cols.keys():
                mean_col_name   = 'user_%s_mean'%type_cols
                median_col_name = 'user_%s_median'%type_cols
                entropy_col_name = 'user_%s_entropy'%type_cols
                gini_col_name = 'user_%s_gini'%type_cols
                sd_col_name     = 'user_%s_sd'%type_cols
                cols = all_cols[type_cols]
                tmp = []
                row = fet1.loc[ind,cols]
                speech_data = []
                if type_cols == 'speech':
                    for col in all_cols[type_cols]:
                        if isinstance(row[col],str):
                            speech_data.append(row[col].split('\'')[1])
                    tmp_df_row['user_speech'] = " ".join(speech_data)
                else:
                    row_np = pk = row.to_numpy()
                    tmp_df_row[mean_col_name] = row_np.mean()
                    tmp_df_row[sd_col_name] = row_np.std()
                    tmp_df_row[gini_col_name]  = gini(row_np)
                    tmp_df_row[entropy_col_name] = entropy((row_np/(row_np.sum()+.0000001)).tolist())

            tmp_df = tmp_df.append(tmp_df_row,ignore_index=True)
        features[group] = tmp_df

    final_features = pd.concat(features.values())
    return final_features

parser = argparse.ArgumentParser()
parser.add_argument('-f','--fileprefix', required=True,
                        help='Specify the file prefix  (until group number appears)for cotrack features file. The script will generate automatically the file names for each group')
parser.add_argument('-s','--session',required=True,
                        help='Specify session number used to generate group names')
parser.add_argument('-n','--num_groups',required=True,type=int,
                        help="Specify number of groups")
args = vars(parser.parse_args())

groups = []
files = []
for ind in range(args['num_groups']):
    group = args['session'] + '_' + str(ind+1)
    file = args['fileprefix'] + str(ind+1) + '.csv'
    groups.append(group)
    files.append(file)
final_file_name = 'Session_' + args['session'] + '_mergeFeatures.csv'
df = mergeFeatures(files,groups)
print(' Total %s instances generate each with %s features'%(df.shape[0],df.shape[1]))
print(' All features are saved in ',final_file_name)
df.to_csv(final_file_name,index=False)
