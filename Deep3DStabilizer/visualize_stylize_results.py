import cv2 as cv
import os
import sys
import glob
import numpy as np
import random

# scenes = [
#     "chess-01",
#     "fire-01",
#     "office-02",
#     "head-01",
#     "redkitchen-04",
#     "stairs-03"
# ]



H, W = 1000, 1280
style_ids = [6,8,10,15,25,28,36,44,73,75]
for style_id in style_ids:
    resLs = sorted(glob.glob(f'stylize_log/{style_id:02}_*_L.png'))
    resRs = sorted(glob.glob(f'stylize_log/{style_id:02}_*_R.png'))
    # rgb_imgs = sorted(glob.glob(f'../preprocess/outputs/{scene}-RGB/*.png'))
    # # print(rgb_imgs)
    # rec_imgs = sorted(glob.glob(f'../preprocess/outputs/{scene}-feat/*.png'))
    # rand_int = random.randint(0,4)
    # sty_imgs = sorted(glob.glob(f'../stylescene/exp/experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_{scene}_0.25_n4/{rand_int:02}_*.jpg'))
    h, w = 480, 640 
    # path_img_style = f"../style_data/style120"

    print('=> The output video will be saved as ./Courtroom_{}.mp4'.format(style_id))
        
    video_writer = cv.VideoWriter(f'./Courtroom_{style_id}.mp4', 0x7634706d, int(30), (W,H))
    img_style = cv.imread(os.path.join('style','{}.jpg'.format(style_id)))
    img_style = cv.resize(img_style, (120,120))

    for j in range(200):
        # if f[-4:] != '.png' and f[-4:] != '.jpg':
        #     continue
        # print(f)
        # d = np.load(depths[j])
        # d *= 25
        # d = d.astype('uint8')
        # d = cv.cvtColor(d, cv.COLOR_GRAY2BGR)
        # d = cv.resize(d, (w,h))
        # rgb_img = cv.imread(rgb_imgs[j])[:h,:w]
        # rec_img = cv.imread(rec_imgs[j])[:h,:w]
        # sty_img = cv.imread(sty_imgs[j])[:h,:w]
        resL = cv.imread(resLs[j])
        resR = cv.imread(resRs[j])

        img_total = np.zeros((H,W,3)).astype(np.uint8)
        # img_total[(H-2*h)//2:(H-2*h)//2+h, :w, :] = d
        img_total[(H-h)//2:(H-h)//2+h, :w, :] = resL
        img_total[(H-h)//2:(H-h)//2+h, w:2*w, :] = resR
        # img_total[(H-h)//2:(H-h)//2+h, w:2*w, :] = rec_img
        img_total[(H-h)//2-120:(H-h)//2, :120, :] = img_style
        # img_total[(H-h)//2-120:(H-h)//2,:120,:] = img_style
        video_writer.write(img_total)