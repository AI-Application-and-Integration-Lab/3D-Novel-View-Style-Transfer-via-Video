import moviepy.editor as mp
import sys
clip = mp.VideoFileClip(sys.argv[1])
clip_resized = clip.resize((640,480)) 
clip_resized.write_videofile(sys.argv[2])