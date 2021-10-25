
from moviepy.editor import *
def concatenate_mkv(file1,file2,output_file_name):
    ts1 = int(file1.split("_")[5].split(".")[0])
    ts2 = int(file2.split("_")[5].split(".")[0])

    if ts1 > ts2:
        first = file2
        second = file1
    else:
        first = file1
        second = file2

    clip1 = VideoFileClip(first)
    clip2 = VideoFileClip(second)

    end1 = ts1/1000
    end2 = ts2/1000

    start1 = end1 - clip1.duration
    start2 = end2 - clip2.duration

    filler_duration = start2 - end1

    filler_clip = ColorClip(size=(640,480),color=(0,0,0),duration=filler_duration)

    clip1 = clip1.resize((640,480))
    clip2 = clip2.resize((640,480))

    clips = [clip1.subclip(0,clip1.duration),filler_clip,clip2.subclip(0,clip2.duration)]

    concatenate_videoclips(clips).write_videofile(output_file_name,fps=24,codec='libx264')
