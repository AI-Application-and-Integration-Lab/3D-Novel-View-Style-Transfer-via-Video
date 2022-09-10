import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import argparse, sys, os, csv, time, datetime
import scipy
from tqdm import tqdm
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation as R
from path import Path
from PIL import Image
# import options

from warper import Warper, inverse_pose
from sequence_io import *
from models.stylize import net, styler, vgg16
from flow import *

from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter as scipy_gaussian
import warnings

import logging
from log_helper import init_log
from models.stylize.utils import repackage_hidden


init_log('global', logging.INFO)
logger = logging.getLogger('global')

parser = argparse.ArgumentParser()

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--bank', type=int, default=0)

args = parser.parse_args()

dirs_test = [
    '7scene-chess-01',
    # '7scene-fire-01',
    # '7scene-head-01',
    # '7scene-office-01',
    # '7scene-office-02',
    # '7scene-redkitchen-01',
    # '7scene-redkitchen-04',
    # '7scene-stairs-03',
]

def style_transform():
    transform_list = [
        trn.Resize(size=(384, 384)),
        trn.ToTensor()
    ]
    return trn.Compose(transform_list)

def content_transform():
    transform_list = [
        # trn.Resize(size=(384, 384)),
        trn.ToTensor()
    ]
    return trn.Compose(transform_list)

class MyDataset(Dataset):
    def __init__(self, path):
        self.image_names_L = sorted(list(glob.glob(path/'images_L/*.png')))
        self.image_names_R = sorted(list(glob.glob(path/'images_R/*.png')))
        self.transform = content_transform()

    def __getitem__(self, index):
        return self.base_getitem(index)

    def __len__(self):
        return self.base_len()

    def base_len(self):
        return len(self.image_names_L) - 1

    def base_getitem(self, index):
        cur_im_L = Image.open(self.image_names_L[index+1]).convert('RGB')
        cur_im_L = self.transform(cur_im_L)
        cur_im_R = Image.open(self.image_names_R[index+1]).convert('RGB')
        cur_im_R = self.transform(cur_im_R)
        prev_im_L = Image.open(self.image_names_L[index]).convert('RGB')
        prev_im_L = self.transform(prev_im_L)
        prev_im_R = Image.open(self.image_names_R[index]).convert('RGB')
        prev_im_R = self.transform(prev_im_R)
        return cur_im_L, cur_im_R, prev_im_L, prev_im_R

class MultiDataset(Dataset):
    def __init__(self, name, *datasets, uniform_sampling=False):
        self.name = name
        self.datasets = []
        self.n_samples = []
        self.cum_n_samples = [0]
        self.uniform_sampling = uniform_sampling

    def append(self, dataset):
        if not isinstance(dataset, MyDataset):
            raise Exception("invalid Dataset in append")
        self.datasets.append(dataset)
        self.n_samples.append(dataset.base_len())
        n_samples = self.cum_n_samples[-1] + dataset.base_len()
        self.cum_n_samples.append(n_samples)

    def __len__(self):
        return self.cum_n_samples[-1]

    def __getitem__(self, idx):
        idx = idx % len(self)
        didx = np.searchsorted(self.cum_n_samples, idx, side="right") - 1
        sidx = idx - self.cum_n_samples[didx]
        return self.datasets[didx].base_getitem(sidx)

def get_test_set(path):
    name = str(path).split("/")[1]
    print(f"  create dataset for {name}")

    dset = MyDataset(path)
    return dset

def get_test_sets():
    # logging.info("Create train datasets")
    dsets = MultiDataset(name="test")
    for path in dirs_test:
      path = Path("outputs") / path
      dsets.append(get_test_set(path))
    return dsets

def styleInput():
    imgs = []
    style_tf = style_transform()
    for i in range(120):
        path = '{}.jpg'.format(i)
        img = Image.open(os.path.join('./style', path)).convert('RGB')
        img = style_tf(img).unsqueeze(0)
        img_arr = []
        for j in range(args.batch_size):
            img_arr.append(img)
        img = torch.cat(img_arr, dim=0)
        imgs.append(img)
    return imgs

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg = vgg16.Vgg16()
    vgg.eval()
    mstyler = styler.ReCoNet()
    mstyler.eval()
    mstyler.load_state_dict(torch.load('weights/120style_noflow_nolstm.tar')['state_dict'], strict=True)
    flow_processor = FlowProcessor(None).to(device)
    network = net.Net(mstyler, vgg=vgg, flow=flow_processor)
    network = network.cuda()
    network.styler = network.styler.cuda()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    content_dataset = get_test_sets()

    content_loader = data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads)

    print('loading dataset done', flush=True)

    style_bank = styleInput()

    content_iter = iter(content_loader)

    for i, data in enumerate(content_loader):
        cur_content_L, cur_content_R, prev_content_L, prev_content_R = data
        cur_content_L, cur_content_R, prev_content_L, prev_content_R = cur_content_L.cuda(), cur_content_R.cuda(), prev_content_L.cuda(), prev_content_R.cuda()
        for bank in range(10):
            style_images = style_bank[bank]
            style_images = Variable(style_images.cuda(), requires_grad=False)
            cur_content_L, cur_content_R, prev_content_L, prev_content_R = cur_content_L.cuda(), cur_content_R.cuda(), prev_content_L.cuda(), prev_content_R.cuda()

            loss_c, loss_s, loss_tv, loss_tc, g_ts = network(cur_content_L, cur_content_R, prev_content_L, prev_content_R, style_images, bank)
            res0 = g_ts[2][0].detach().cpu().numpy().transpose(1,2,0)
            res0 = Image.fromarray((res0*255).astype(np.uint8))
            res0.save(f'stylize_log/{bank:02}_{i:06}_0.png')
    


