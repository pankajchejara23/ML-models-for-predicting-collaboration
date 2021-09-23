"""
This script is for combining fragments of webm files collected using CoTrack. Though CoTrack store final webm file
sometimes the final file may be missing. In such cases, this script will combine all fragments for video
recording for each user and creates a single file for later analysis.
"""

from os import walk
import functools
import os
import sys
# function to be used for sorting file list on the basis of timestamp
def compare(item1,item2):
    item1 = item1.split('.')[0]
    item2 = item2.split('.')[0]
    i1 = int(item1.split('_')[4])
    i2 = int(item2.split('_')[4])

    if i1 > i2:
        return 1
    else:
        return -1

dirlist= []

if len(sys.argv) < 3:
    print('Incorrect use, please use the script in following way.\n')
    print("Use this format: \n python files_merge.py <source_direcotyr> <target_directory>")
    print('\n here <source_directory> is cotrack directory from media folder. This directory contains group-wise video recordings.')
    print('        <target_directory> is the directory where you want to save the merged files for each participants in the group.\n\n')
    exit()
else:
    source_directory = sys.argv[1]
    target_directory = sys.argv[2]
    print('Directory:',source_directory)


if not os.path.isdir(target_directory):
    os.makedirs(target_directory)


for (dirpath, dirnames, filenames) in walk(source_directory):
    dirlist.append(dirpath)

# taking only webm files
flist = [f for f in dirlist if 'user_' in f]

print('\n'.join(flist))
for d in flist:
    uflist = []
    print(d)
    for f in os.listdir(d):

        if len(f.split('_')) == 5:
            uflist.append(f)
    uflist.sort(key=functools.cmp_to_key(compare))
    #print(' \n'.join(uflist))

    timestamp = ''
    first_file = True
    for file in uflist[::-1]:
        if first_file:
            timestamp = file.split('_')[4]
            print('   Timestamp:',timestamp)
            break

    info = d.split('/')

    session = info[-3].split('_')[1]
    group = info[-2].split('_')[1]
    user = info[-1].split('_')[1]

    final = target_directory + '/' + '_'.join([session,group,user,'Final_file',timestamp])
    print('Output file:',final)
    out_file = open(final,'wb')
    for file in uflist:
        #print('..writing ',file)
        file = d +'/'+ file

        in_file = open(file,'rb')
        out_file.write(in_file.read())
