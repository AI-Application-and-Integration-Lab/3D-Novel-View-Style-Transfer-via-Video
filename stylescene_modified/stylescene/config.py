from pathlib import Path
import socket
import platform
import getpass

Train = False
Test_style = [i for i in range(120)]
            #[0,2,3,9,14,20,45,46,71,119] #00,03,09,10,46 good
                               #02,08,17,28 boring
                               #04,5,11,14,15,25,27,50,120,38 not good

tat_root = Path("../../colmap_tat/")
styleroot = "../../style_data/"
HOSTNAME = socket.gethostname()
PLATFORM = platform.system()
USER = getpass.getuser()
train_device = "cuda:0"
eval_device = "cuda:0"
dtu_root = None
colmap_bin_path = None
lpips_root = None

frames_per_pc = 1
snippet_size = 17
snippet_stride = 1
use_local_pointcloud = True # Set False for original stylescene
use_naive = False


tat_train_sets = [
    # "training/Barn",
    # "training/Caterpillar",
    # "training/Church",
    # "training/Ignatius",
    # "training/Meetingroom",
    # "intermediate/Family",
    # "intermediate/Francis",
    # "intermediate/Horse",
    # "intermediate/Lighthouse",
    # "intermediate/Panther",
    # "advanced/Auditorium",
    # "advanced/Ballroom",
    # "advanced/Museum",
    # "advanced/Temple",
    # "advanced/Courtroom",
    # "advanced/Palace",
    "fire-01",
    "fire-03",
    "office-01",
    "office-02",
    "redkitchen-04",
    "redkitchen-01",
    "head-01",
    "head-02",
    "chess-01",
    "chess-02",
    "stairs-03",
    "stairs-02",
    "pumpkin-02",
    "pumpkin-06"
]

tat_eval_sets = [
    # "training/Truck",
    # "intermediate/M60",
    # "intermediate/Playground",
    # "intermediate/Train",
    # "Playground",
    # "Ignatius",
    # "Courtroom",
    "Truck",
    # "Train",
    # "Horse",
    # "Auditorium",
    # "Family",
    # "Francis",
    # "Meetingroom",
    # "Palace",
    # "Panther",
    # "Lighthouse",
    # "fire-01",
    # "office-01",
    # "office-02",
    # "redkitchen-04",
    # "head-01",
    # "chess-01",
    # "stairs-03",
    # "pumpkin-02",
    # "fire-01-col",
    # "office-01-col",
    # "redkitchen-04-col",
    # "head-01-col",
    # "chess-01-col",
    # "stairs-03-col",
    # "Playground-col",
    # "Ignatius-col",
    # "Courtroom-col",
    # "Truck-col",
    # "Train-col",
    # "Horse-col",
    # "Auditorium-col",
    # "Family-col",
    # "Francis-col",
    # "Meetingroom-col",
    # "Palace-col",
    # "Panther-col",
]

tat_eval_tracks = {}
# tat_eval_tracks['training/Truck'] = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
# tat_eval_tracks['intermediate/M60'] = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
# tat_eval_tracks['intermediate/Playground'] = [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
# tat_eval_tracks['intermediate/Train'] = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]
tat_eval_tracks['office-01'] = [i for i in range(500)]
tat_eval_tracks['fire-01'] = [i for i in range(500)]
tat_eval_tracks['office-02'] = [i for i in range(500)]
tat_eval_tracks['redkitchen-04'] = [i for i in range(500)]
tat_eval_tracks['head-01'] = [i for i in range(500)]
tat_eval_tracks['chess-01'] = [i for i in range(500)]
tat_eval_tracks['stairs-03'] = [i for i in range(500)]
tat_eval_tracks['pumpkin-02'] = [i for i in range(500)]
tat_eval_tracks['fire-01-col'] = [i for i in range(500)]
tat_eval_tracks['office-01-col'] = [i for i in range(500)]
tat_eval_tracks['redkitchen-04-col'] = [i for i in range(500)]
tat_eval_tracks['head-01-col'] = [i for i in range(500)]
tat_eval_tracks['chess-01-col'] = [i for i in range(500)]
tat_eval_tracks['stairs-03-col'] = [i for i in range(500)]
tat_eval_tracks['Playground'] = [i for i in range(500)]
tat_eval_tracks['Ignatius'] = [i for i in range(500)]
tat_eval_tracks['Courtroom'] = [i for i in range(500)]
tat_eval_tracks['Truck'] = [i for i in range(500)]
tat_eval_tracks['Train'] = [i for i in range(500)]
tat_eval_tracks['Horse'] = [i for i in range(500)]
tat_eval_tracks['Auditorium'] = [i for i in range(500)]
tat_eval_tracks['Family'] = [i for i in range(500)]
tat_eval_tracks['Francis'] = [i for i in range(500)]
tat_eval_tracks['Meetingroom'] = [i for i in range(500)]
tat_eval_tracks['Palace'] = [i for i in range(500)]
tat_eval_tracks['Panther'] = [i for i in range(500)]
tat_eval_tracks['Lighthouse'] = [i for i in range(500)]
tat_eval_tracks['Ignatius-col'] = [i for i in range(500)]
tat_eval_tracks['Courtroom-col'] = [i for i in range(500)]
tat_eval_tracks['Truck-col'] = [i for i in range(500)]
tat_eval_tracks['Train-col'] = [i for i in range(500)]
tat_eval_tracks['Horse-col'] = [i for i in range(500)]
tat_eval_tracks['Playground-col'] = [i for i in range(500)]
tat_eval_tracks['Auditorium-col'] = [i for i in range(100,300)]
tat_eval_tracks['Family-col'] = [i for i in range(100,300)]
tat_eval_tracks['Francis-col'] = [i for i in range(100,300)]
tat_eval_tracks['Meetingroom-col'] = [i for i in range(100,300)]
tat_eval_tracks['Palace-col'] = [i for i in range(100,300)]
tat_eval_tracks['Panther-col'] = [i for i in range(500)]