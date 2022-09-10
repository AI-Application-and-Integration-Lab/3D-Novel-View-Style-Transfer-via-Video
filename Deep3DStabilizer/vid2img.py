import numpy as np
import cv2
import os
input_loc = 'Museum.mp4'
output_loc = 'tmp_images'

if not os.path.exists(output_loc):
    os.mkdir(output_loc)
cap = cv2.VideoCapture(input_loc)
# Find the number of frames
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print ("Number of frames: ", video_length)
count = 0
print ("Converting video..\n")
# Start converting the video
while cap.isOpened():
    # Extract the frame
    ret, frame = cap.read()
    if not ret:
        continue
    # Write the results back to output location.
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    if count % 2 == 0:
        cv2.imwrite(output_loc + "/%#05d.png" % (count//2), frame)
    count = count + 1
    # If there are no more frames left
    if (count > min(2000, (video_length-1))):
        # Log the time again
        # Release the feed
        cap.release()
        # Print stats
        print ("Done extracting frames.\n%d frames extracted" % count)
        break