This is the instruction of running "3D Novel View Style Transfer via Learning-based Video Depth Estimation"

Step 1.
Download 'mask_rcnn_R_50_FPN_3x.pkl' from "https://drive.google.com/file/d/1HPgo1-zBv29UGii5UvizTHaW7c8U06dR/view?usp=sharing" and put it in 'Deep3DStabilizer/weights'

Step 2.
Generate 3D from "Deep3DStabilizer", the instructions can be refer to "Readme.docx" in the directory.

Step 3.
Transform the coordinates of camera poses by running "move_to_ibr3d.py" in './stylescene_modified/colmap_tat':

python move_to_ibr3d.py ${your_output_directory}

Your transformed informations will be add to the 'ibr3d_pw_0.25' directory in your output directory.

Step 4.
Create your dataset directory in './stylescene_modified/colmap_tat' and move the 'ibr3d_pw_0.25' to this directory. Details can be refered to the example dataset in './stylescene_modified/colmap_tat'.

Step 5.
Run the installation of stylescene_modified. The instruction is the same as 'stylescene' proposed by [Huang et al.].

Step 6.
Run 'python create_data_pw_tat.py -s 0.25' in './FreeViewSynthesis_modified/data'. The path of dataset can be refered in './FreeViewSynthesis_modified/config.py'. Remember to copy a folder named 'ibr3d_long'.

Step 7.
'cd stylescene_modified/preprocess/'

Step 8.
'python Get_feat.py'

Step 9.
'cd ../stylescene/exp'

Step 10.
Download pretrained model of stylization modules at "https://drive.google.com/file/d/1s5BOENQHOZv9qbLo1GFX5rFJO09nxtV1/view?usp=sharing" and put it in "experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3" folder.

Step 11.
'bash test_own.sh', the "--eval-style-id" parameter in 'test_own.sh can be set by users.
