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
from models.stylize import lstmnet, styler2, vgg16
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
parser.add_argument('--name', default='7scene-stairs-03',
                    help='Directory of input images')
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
    # '7scene-chess-01',
    # '7scene-fire-01',
    # '7scene-head-01',
    # '7scene-office-01',
    # '7scene-office-02',
    # '7scene-redkitchen-01',
    # '7scene-redkitchen-04',
    # '7scene-stairs-03',
    # '7scene-pumpkin-02',
    # 'Ignatius',
    # 'Playground',
    # 'Courtroom',
    # 'market_1',
    # 'mountain_2',
    args.name
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

def vgg_norm(var):
    dtype = torch.cuda.FloatTensor
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    normed = var.sub(mean).div(std)
    return normed

class MyDataset(Dataset):
    def __init__(self, path):
        self.image_names_L = sorted(list(glob.glob(path/'images_L/*.png')))[:250]
        self.image_names_R = sorted(list(glob.glob(path/'images_R/*.png')))[:250]
        self.transform = content_transform()

    def __getitem__(self, index):
        return self.base_getitem(index)

    def __len__(self):
        return self.base_len()

    def base_len(self):
        return len(self.image_names_L)

    def base_getitem(self, index):
        
        im_L = Image.open(self.image_names_L[index]).convert('RGB')
        im_L = self.transform(im_L)
        im_R = Image.open(self.image_names_R[index]).convert('RGB')
        im_R = self.transform(im_R)
            
        return im_L, im_R

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
    mstyler = styler2.ReCoNet().cuda()
    mstyler.eval()
    model_checkpoint = torch.load('weights/120style_LRvideo.pth.tar')['state_dict']
    # model_checkpoint = torch.load('weights/120style_noflow_nolstm.tar')
    
    mstyler.load_state_dict(model_checkpoint, strict=True)
    flow_processor = FlowProcessor(None).to(device)
    network = lstmnet.VideoNet(mstyler, vgg=vgg, flow=flow_processor)
    network = network.cuda()
    network.styler = network.styler.cuda()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(os.path.join('stylize_log', args.name)):
        os.mkdir(os.path.join('stylize_log', args.name))

    if not os.path.exists(os.path.join('stylize_log', args.name, 'images_L')):
        os.mkdir(os.path.join('stylize_log', args.name, 'images_L'))

    if not os.path.exists(os.path.join('stylize_log', args.name, 'images_R')):
        os.mkdir(os.path.join('stylize_log', args.name, 'images_R'))

    content_dataset = get_test_sets()

    content_loader = data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads)

    print('loading dataset done', flush=True)

    style_bank = styleInput()

    # content_iter = iter(content_loader)
    temporal_time = 1
    average_time = 0.0

    for bank in range(21,22): # [0,2,9,14,19,26,28,39,40,46,52,71,92,107,119]
        print(bank)
        if not os.path.exists(os.path.join('stylize_log', args.name, 'images_L', str(bank))):
            os.mkdir(os.path.join('stylize_log', args.name, 'images_L', str(bank)))
        if not os.path.exists(os.path.join('stylize_log', args.name, 'images_R', str(bank))):
            os.mkdir(os.path.join('stylize_log', args.name, 'images_R', str(bank)))
        style = style_bank[bank].cuda()
        prev_state1 = None
        prev_state2 = None
        prev_content_L = []
        prev_content_R = []
        prev_stylize_L = []
        prev_stylize_R = []
        content_iter = iter(content_loader)
        n = len(content_loader)
        for i in range(len(content_loader)):
            start_time = time.time()
            data = next(content_iter)
            cur_content_L, cur_content_R = data
            # print(cur_content_L.shape)
            cur_content_L = cur_content_L.cuda()
            cur_content_R = cur_content_R.cuda()
        
            with torch.no_grad():
                if i < temporal_time:
                    outL, outR, prev_state1, prev_state2, maskL, maskR, prevL_warp, prevR_warp = network.forward_eval(\
                        cur_content_L, cur_content_R, None, None,\
                        style, prev_state1, prev_state2, bank, None, None, False)
                else:
                    outL, outR, prev_state1, prev_state2, maskL, maskR, prevL_warp, prevR_warp = network.forward_eval(\
                        cur_content_L, cur_content_R, prev_content_L[0], prev_content_R[0],\
                        style, prev_state1, prev_state2, bank, prev_stylize_L[0], prev_stylize_R[0], False)
        
            prev_state1 = repackage_hidden(prev_state1)
            prev_state2 = repackage_hidden(prev_state2)
            if len(prev_content_L) == temporal_time:
                prev_content_L.pop(0)
                prev_content_R.pop(0)
                prev_stylize_L.pop(0)
                prev_stylize_R.pop(0)
            prev_content_L.append(cur_content_L.detach())
            prev_content_R.append(cur_content_R.detach())
            prev_stylize_L.append(outL.detach())
            prev_stylize_R.append(outR.detach())
            res0 = outL[0].cpu().numpy().transpose(1,2,0)
            res0 = Image.fromarray((res0*255).astype(np.uint8))
            res0.save(f'stylize_log/{args.name}/images_L/{bank}/{i:06}.png')
            res1 = outR[0].detach().cpu().numpy().transpose(1,2,0)
            res1 = Image.fromarray((res1*255).astype(np.uint8))
            res1.save(f'stylize_log/{args.name}/images_R/{bank}/{i:06}.png')
            end_time = time.time()
            print(end_time - start_time)
            average_time += (end_time - start_time)
            # if maskL is not None and maskR is not None:
            #     mask0 = maskL[0][0].detach().cpu().numpy()
            #     mask0 = Image.fromarray((mask0*255).astype(np.uint8))
            #     mask0.save(f'mask_log/{bank:02}_{i:06}_L.png')
            #     # mask1 = maskR[0][0].detach().cpu().numpy()
            #     # mask1 = Image.fromarray((mask1*255).astype(np.uint8))
            #     # mask1.save(f'mask_log/{bank:02}_{i:06}_R.png')
            # if prevL_warp is not None and prevR_warp is not None:
            #     prev0 = prevL_warp[0].detach().cpu().numpy().transpose(1,2,0)
            #     prev0 = Image.fromarray((prev0*255).astype(np.uint8))
            #     prev0.save(f'stylize_log/{args.name}/images_L/{bank}/{i-temporal_time:06}_L_w.png')
        print(average_time / n)


