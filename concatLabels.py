import pandas as pd
from scipy.stats import entropy
import math
import numpy as np
import argparse

def concatLabels(files,groups):
    label_columns = ['frame','SMU', 'ARG', 'KE', 'CF', 'CO', 'STR']
    all_labels = []
    for index,file in enumerate(files):
        label_df = pd.DataFrame(columns = label_columns+['group','ITO'])
        group = groups[index]
        fet1 = pd.read_csv(file)
        print(fet1.shape)
        fet1.frame = fet1.index
        ITO_cols = []
        for col in fet1.columns:
            sp = col.split('_')
            if len(sp) == 2 and sp[0] == 'ITO':
                ITO_cols.append(col)
        for ind in range(fet1.shape[0]):
            row = fet1.loc[ind,:]
            tmp_df_row = {}
            tmp_df_row['group'] = group
            for col in label_columns:
                tmp_df_row[col] = row[col]
            ito_row = fet1.loc[ind,ITO_cols].to_numpy()
            tmp_df_row['ITO'] = ito_row.mean()
            print(tmp_df_row)
            label_df = label_df.append(tmp_df_row,ignore_index=True)
        all_labels.append(label_df)
    final_df = pd.concat(all_labels)
    return final_df

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
final_file_name = 'session_' + args['session'] + '_all_labels.csv'
df = concatLabels(files,groups)
print(' Total %s labels generate each with %s dimensions'%(df.shape[0],df.shape[1]))
print(' All features are saved in ',final_file_name)
df.to_csv(final_file_name,index=False)
