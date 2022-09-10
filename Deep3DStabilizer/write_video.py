import cv2 as cv
import os
import sys

print('=> The output video will be saved as ./{}.mp4'.format(sys.argv[1][:-1]))
f = list(sorted(os.listdir(sys.argv[1])))[0]
img = cv.imread(os.path.join(sys.argv[1], f))
H, W, _ = img.shape
# H, W = 480, 640
video_writer = cv.VideoWriter('./{}.mp4'.format(sys.argv[1][:-1]), 0x7634706d, int(30), (W,H))
# print(sorted(os.listdir(sys.argv[1])))
for f in sorted(os.listdir(sys.argv[1])):
    if f[-4:] != '.png' and f[-4:] != '.jpg':
        continue
    # print(f)
    img = cv.imread(os.path.join(sys.argv[1], f))
    # img = cv.resize(img, (W,H))
    video_writer.write(img)