# python geometry_optimizer.py videos/Auditorium-640.mp4 --name Auditorium
# rm -rf outputs/Auditorium/flows

# python rectify.py videos/Auditorium-640.mp4 --name Auditorium

# python test_stylize_LR_video.py --name Auditorium

# python test_stylize_LR_video.py --name Courtroom
# python test_stylize_LR_video.py --name Horse
# python test_stylize_LR_video.py --name Ignatius
# python test_stylize_LR_video.py --name Playground
# python test_stylize_LR_video.py --name Truck
# python test_stylize_LR_video.py --name Train

# python calc_lpips.py Courtroom Gao
# python calc_lpips.py Courtroom-col 0
# python calc_lpips.py Courtroom 50000
# python calc_lpips.py Courtroom 79999

python calc_L2.py Courtroom Gao
python calc_L2.py Courtroom 0
python calc_L2.py Courtroom 50000
python calc_L2.py Courtroom 79999
# python calc_L2.py Horse Gao
# python calc_L2.py Horse 0
# python calc_L2.py Horse 50000
# python calc_L2.py Horse 79999
# python calc_L2.py Ignatius Gao
# python calc_L2.py Ignatius 0
# python calc_L2.py Ignatius 50000
# python calc_L2.py Ignatius 79999
# python calc_L2.py Playground Gao
# python calc_L2.py Playground 0
# python calc_L2.py Playground 50000
# python calc_L2.py Playground 79999
python calc_L2.py Truck Gao
python calc_L2.py Truck 0
python calc_L2.py Truck 50000
python calc_L2.py Truck 79999
python calc_L2.py Train Gao
python calc_L2.py Train 0
python calc_L2.py Train 50000
python calc_L2.py Train 79999
# python calc_L2.py Auditorium Gao
# python calc_L2.py Auditorium 0
# python calc_L2.py Auditorium 50000
# python calc_L2.py Auditorium 79999
# python calc_L2.py Family Gao
# python calc_L2.py Family 0
# python calc_L2.py Family 50000
# python calc_L2.py Family 79999
# python calc_L2.py Francis Gao
# python calc_L2.py Francis 0
# python calc_L2.py Francis 50000
# python calc_L2.py Francis 79999
# python calc_L2.py Meetingroom Gao
# python calc_L2.py Meetingroom 0
# python calc_L2.py Meetingroom 50000
# python calc_L2.py Meetingroom 79999
# python calc_L2.py Palace Gao
# python calc_L2.py Palace 0
# python calc_L2.py Palace 50000
# python calc_L2.py Palace 79999
python calc_L2.py Panther Gao
python calc_L2.py Panther 0
python calc_L2.py Panther 50000
python calc_L2.py Panther 79999

# python calc_L2.py Courtroom ReRe
# python calc_L2.py Horse ReRe
# python calc_L2.py Ignatius ReRe
# python calc_L2.py Playground ReRe
# python calc_L2.py Truck ReRe
# python calc_L2.py Train ReRe
# python calc_L2.py Auditorium ReRe
# python calc_L2.py Family ReRe
# python calc_L2.py Francis ReRe
# python calc_L2.py Meetingroom ReRe
# python calc_L2.py Palace ReRe
# python calc_L2.py Panther ReRe

# python calc_L2.py Courtroom 1
# python calc_L2.py Courtroom-col 1
# python calc_L2.py Horse 1
# python calc_L2.py Horse-col 1
# python calc_L2.py Ignatius 1
# python calc_L2.py Ignatius-col 1
# python calc_L2.py Playground 1
# python calc_L2.py Playground-col 1
# python calc_L2.py Truck 1
# python calc_L2.py Truck-col 1
# python calc_L2.py Train 1
# python calc_L2.py Train-col 1


# ffmpeg -ss 0 -i Ignatius.mp4 -t 17 -c copy Ignatius-sub.mp4

# python video_resizer.py Ignatius-sub.mp4 Ignatius-640.mp4

# python calc_lpips.py stylize_log/Truck/
# python calc_lpips.py stylize_log/7scene-chess-01/

# python rectify.py videos/7scene-chess-01.mp4 --name 7scene-chess-01_col
# python rectify.py videos/7scene-fire-01.avi --name 7scene-fire-01_col
# python rectify.py videos/head-01.avi --name 7scene-head-01_col
# python rectify.py videos/office-02.avi --name 7scene-office-02_col
# python rectify.py videos/pumpkin-02.mp4 --name 7scene-pumpkin-02_col
# python rectify.py videos/redkitchen-04.avi --name 7scene-redkitchen-04_col
# python rectify.py videos/stairs-03.avi --name 7scene-stairs-03_col

# python outputs/move_to_ibr3d.py outputs/7scene-chess-01
# python outputs/move_to_ibr3d.py outputs/7scene-chess-02
# python outputs/move_to_ibr3d.py outputs/7scene-fire-01
# python outputs/move_to_ibr3d.py outputs/7scene-fire-03
# python outputs/move_to_ibr3d.py outputs/7scene-head-01
# python outputs/move_to_ibr3d.py outputs/7scene-head-02
# python outputs/move_to_ibr3d.py outputs/7scene-office-01
# python outputs/move_to_ibr3d.py outputs/7scene-office-02
# python outputs/move_to_ibr3d.py outputs/7scene-pumpkin-02
# python outputs/move_to_ibr3d.py outputs/7scene-pumpkin-06
# python outputs/move_to_ibr3d.py outputs/7scene-redkitchen-01
# python outputs/move_to_ibr3d.py outputs/7scene-redkitchen-04
# python outputs/move_to_ibr3d.py outputs/7scene-stairs-02
# python outputs/move_to_ibr3d.py outputs/7scene-stairs-03