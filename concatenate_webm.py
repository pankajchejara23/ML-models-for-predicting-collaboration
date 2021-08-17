"""
This script is for combining fragments of webm files collected using CoTrack. Though CoTrack store final webm file
sometimes the final file may be missing. In such cases, this script will combine all fragments for video
recording for each user and creates a single file for later analysis.

"""

from os import walk
import functools
import os

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
for (dirpath, dirnames, filenames) in walk("."):
    dirlist.append(dirpath)

# taking only webm files
flist = [f for f in dirlist if len(f.split('/'))==4]

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
    for file in uflist:
        if first_file:
            timestamp = file.split('_')[4]
            print('   Timestamp:',timestamp)
            break

    final = d.replace('.','').replace('/','_') + '_' + str(timestamp)
    out_file = open(final,'wb')
    for file in uflist:
        #print('..writing ',file)
        file = d +'/'+ file
        in_file = open(file,'rb')
        out_file.write(in_file.read())
