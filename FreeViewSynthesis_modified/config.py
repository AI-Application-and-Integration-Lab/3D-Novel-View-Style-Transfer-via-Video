from pathlib import Path
import socket
import platform
import getpass

HOSTNAME = socket.gethostname()
PLATFORM = platform.system()
USER = getpass.getuser()

train_device = "cuda:0"
eval_device = "cuda:0"
dtu_root = None
tat_root = None
colmap_bin_path = None
lpips_root = None


tat_train_sets = [
    # "training/Barn",
    # "training/Caterpillar",
    # "training/Church",
    # "training/Courthouse",
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
    # "eth_delivery",
    # "eth_electro",
    # "eth_terrians",
    # "7scene-office-02",
    # "7scene-pumpkin-02",
]

tat_eval_sets = [
    # "training/Truck",
    # "intermediate/M60",
    # "intermediate/Playground",
    # "intermediate/Train",
    # "chess-01",
    # "chess-02",
    # "fire-01",
    # "fire-03",
    # "head-01",
    # "head-02",
    # "office-01",
    # "office-02",
    # "pumpkin-02",
    # "pumpkin-06",
    # "redkitchen-01",
    # "redkitchen-04",
    # "stairs-02",
    # "stairs-03",
    # "Playground",
    # "Ignatius",
    # "Courtroom",
    # "Truck",
    # "Train",
    # "Horse",
    # "Lighthouse",
    # "Auditorium",
    # "Family",
    # "Francis",
    # "Meetingroom",
    # "Palace",
    # "Panther",
    # "Playground-col",
    # "Ignatius-col",
    # "Courtroom-col",
    # "Truck-col",
    # "Train-col",
    # "Horse-col",
    # "Lighthouse-col",
    "Auditorium-col",
    "Family-col",
    "Francis-col",
    "Meetingroom-col",
    "Palace-col",
    "Panther-col",
]

# fmt: off
tat_eval_tracks = {}
# tat_eval_tracks['training/Truck'] = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
# tat_eval_tracks['intermediate/M60'] = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
# tat_eval_tracks['intermediate/Playground'] = [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
# tat_eval_tracks['intermediate/Train'] = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]

dtu_eval_sets = [65, 106, 118]

dtu_interpolation_ind = [13, 14, 15, 23, 24, 25]
dtu_extrapolation_ind = [0, 4, 38, 48]
dtu_source_ind = [idx for idx in range(49) if idx not in dtu_interpolation_ind + dtu_extrapolation_ind]
# fmt: on


lpips_root = None

# TODO: adjust path
tat_root = Path("../../stylescene/colmap_tat/")
