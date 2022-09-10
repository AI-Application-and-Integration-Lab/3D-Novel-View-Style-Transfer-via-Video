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
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=2e5)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--short_weight', type=float, default=100.0)
parser.add_argument('--long_weight', type=float, default=100.0)
parser.add_argument('--tv_weight', type=float, default=0.001)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=2000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--bank', type=int, default=0)

args = parser.parse_args()

dirs_train = [
    '7scene-chess-01', #exist
    '7scene-chess-02', #exist
    '7scene-fire-01', #exist
    '7scene-fire-03',
    '7scene-head-01', #exist
    '7scene-head-02',
    '7scene-office-01', #exist
    '7scene-office-02', #exist
    '7scene-redkitchen-01', #exist
    '7scene-redkitchen-04', #exist
    '7scene-stairs-02',
    '7scene-stairs-03', #exist
    '7scene-pumpkin-02', 
    '7scene-pumpkin-06',
]

def style_transform():
    transform_list = [
        trn.Resize(size=(384, 384)),
        trn.ToTensor()
    ]
    return trn.Compose(transform_list)

def content_transform():
    transform_list = [
        trn.Resize(size=(256, 256)),
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
        return len(self.image_names_L) - 9

    def base_getitem(self, index):
        imgs_L, imgs_R = [], []
        for i in range(10):
            im_L = Image.open(self.image_names_L[index+i]).convert('RGB')
            im_L = self.transform(im_L).unsqueeze(0)
            im_R = Image.open(self.image_names_R[index+i]).convert('RGB')
            im_R = self.transform(im_R).unsqueeze(0)
            imgs_L.append(im_L)
            imgs_R.append(im_R)
        imgs_L = torch.cat(imgs_L, dim=0)
        imgs_R = torch.cat(imgs_R, dim=0)
        return imgs_L, imgs_R

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

def get_train_set(path):
    name = str(path).split("/")[1]
    print(f"  create dataset for {name}")

    dset = MyDataset(path)
    return dset

def get_train_sets():
    # logging.info("Create train datasets")
    dsets = MultiDataset(name="train")
    for path in dirs_train:
      path = Path("outputs") / path
      dsets.append(get_train_set(path))
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

def run():
    # seq_io = SequenceIO(opt, preprocess=False)
    flow_processor = FlowProcessor(None).to(device)

    mstyler = styler2.ReCoNet()
    mstyler.load_state_dict(torch.load('weights/120style.pth.tar'), strict=True)
    mstyler.train()
    vgg = vgg16.Vgg16()
    vgg.eval()
    network = lstmnet.VideoNet(mstyler, vgg=vgg, flow=flow_processor)
    network = network.cuda()
    network.styler.train()
    network.set_train()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    content_dataset = get_train_sets()

    content_loader = data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads)

    print('loading dataset done', flush=True)

    style_bank = styleInput()

    optimizer = torch.optim.Adam(network.styler.parameters(), lr=args.lr, weight_decay=5e-4)

    content_iter = iter(content_loader)
    t0 = time.time()

    for i in range(800000):
        bank = np.random.randint(120)
        style = style_bank[bank]
        prev_state1 = None
        prev_state2 = None
        try:
            inputs = next(content_iter)
        except StopIteration:
            content_iter = iter(content_loader)
            inputs = next(content_iter)
        contents, _ = inputs

        frame_i = []
        frame_o = []
        for t in range(10):
            frame_i.append(contents[:, t, :, :, :])
        style = Variable(style.cuda(), requires_grad=False)
        if style.shape[0] != contents.shape[0]:
            style = style[:contents.shape[0]]

        loss = 0
        for t1 in range(9):
            loss_c, loss_s, loss_t, out, prev_state1, prev_state2 = network(frame_i[t1].to(device), frame_i[t1+1].to(device), style.to(device), prev_state1, prev_state2, bank)
            prev_state1 = repackage_hidden(prev_state1)
            prev_state2 = repackage_hidden(prev_state2)

            frame_o.append(out.detach())
            loss_c = loss_c * args.content_weight
            loss_s = loss_s * args.style_weight
            loss_t = loss_t * args.short_weight
            loss = (loss_c + loss_s + loss_t)
            for t2 in range(1, t1-4):
                loss_t = network.temporal_loss(frame_i[t2].to(device), frame_i[t1+1].to(device), frame_o[t2-1].to(device), out.to(device))
                loss_t = loss_t * args.long_weight / (t1-5)
                loss += loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time.time()
        if (i + 1) % 100 == 0:
            for j in range(9):
                out = frame_o[j][0].detach().cpu().numpy().transpose(1,2,0)
                out = Image.fromarray((out*255).astype(np.uint8))
                out.save(f'stylize_log/{i:06}_{j}.png')
            style_img = style[0].detach().cpu().numpy().transpose(1,2,0)
            style_img = Image.fromarray((style_img*255).astype(np.uint8))
            style_img.save(f'stylize_log/{i:06}_style.png')
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            torch.save({'state_dict': network.styler.state_dict()}, '{:s}/model_iter_{:d}.pth.tar'.format(args.save_dir, i + 1))
        if (i + 1) % 20 == 0:
            logger.info('Iter: [%d] LR:%f Time: %.3f Loss: %.5f LossContet: %.5f  LossSytle: %.5f LossTemporal: %.5f' % 
            (i+1, args.lr, t2 - t0, loss.data.cpu().item(), loss_c.data.cpu().item(), loss_s.data.cpu().item(), loss_t.data.cpu().item()))
            t0 = t2


if __name__ == '__main__':
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    run()