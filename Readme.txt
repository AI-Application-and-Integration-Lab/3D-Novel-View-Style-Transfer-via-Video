This is the instruction of running "3D Novel View Style Transfer via Learning-based Video Depth Estimation"

Step 1.
Generate 3D from "Deep3DStabilizer", the instructions can be refer to "Readme.docx" in the directory.

Step 2.
Transform the coordinates of camera poses by running "move_to_ibr3d.py" in './stylescene_modified/colmap_tat':

python move_to_ibr3d.py ${your_output_directory}

Your transformed informations will be add to the 'ibr3d_pw_0.25' directory in your output directory.

Step 3.
Create your dataset directory in './stylescene_modified/colmap_tat' and move the 'ibr3d_pw_0.25' to this directory. Details can be refered to the example dataset in './stylescene_modified/colmap_tat'.

Step 4.
Run the installation of stylescene_modified. The instruction is the same as 'stylescene' proposed by [Huang et al.]

Step 5.
Run 'python create_data_pw_tat.py -s 0.25' in './FreeViewSynthesis_modified/data'. The path of dataset can be refered in './FreeViewSynthesis_modified/config.py'. Remember to copy a folder named 'ibr3d_long'.

Step 6.
'cd stylescene_modified/preprocess/'

Step 7.
'python Get_feat.py'

Step 8.
'cd ../stylescene/exp'

Step 9.
'bash test_own.sh', the "--eval-style-id" parameter in 'test_own.sh can be set by users.