import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule, find_free_port
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
#from dataloader_cifar import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from mlp import MLP
import unet_1d
from math import ceil
import dataloader_gclemb
import utils
from ema_pytorch import EMA

def sample(params:argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    device = torch.device("cuda", 0)
    
    # load models
    if params.nettype == "unet":    
        net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    elif params.nettype == "unet_1d":
        net = unet_1d.Unet_1d(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    elif params.nettype == "mlp":
        net = MLP(input_size=params.input_size, 
                  hidden_sizes=params.hidden_sizes, 
                  cond_sizes=64, 
                  t_sizes=64,
                  res= params.res)
    else:
        raise NotImplementedError

    cemblayer = ConditionalEmbedding(params.clsnum, params.cdim, params.cdim).to(device)
    # load checkpoints
    checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{params.epoch}_checkpoint.pt'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    cemblayer.load_state_dict(checkpoint['cemblayer'])

    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    ema = EMA(diffusion, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    # start to sample
    #cnt = torch.cuda.device_count()
    cnt = 1
    diffusion.model.eval()
    cemblayer.eval()
    ema.ema_model.eval()
    # generating samples
    numloop = ceil(params.genum  / params.genbatch)
    all_samples = []
    all_labels = []
    each_device_batch = params.genbatch // cnt
    lab = torch.ones(params.clsnum, each_device_batch // params.clsnum).type(torch.long) \
        * torch.arange(start = 0, end = params.clsnum).reshape(-1, 1)
    lab = lab.reshape(-1, 1).squeeze()
    lab = lab.to(device)
    cemb = cemblayer(lab)
    for _ in range(numloop):
        genshape = (each_device_batch ,params.inch, params.inputsize)     #only for 1d data
        if params.ddim:
            #generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
            generated = ema.ema_model.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
        else:
            #generated = diffusion.sample(genshape, cemb = cemb)
            generated  = ema.ema_model.sample(genshape, cemb = cemb)
        # gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
        # all_gather(gathered_samples, img)
        # all_samples.extend([img for img in gathered_samples])
        # all_samples.extend(generated)
        # all_labels.extend(lab)
        all_samples.append(generated)
        all_labels.append(lab)
        
    samples = torch.concat(all_samples, dim = 0)
    labels = torch.concat(all_labels, dim = 0)
    samples = samples.squeeze(1).cpu().numpy()
    labels = labels.cpu().numpy()

    #denormalize
    _, dataset = dataloader_gclemb.load_data(params)
    gt_embs = dataset.emb.squeeze(1).cpu().numpy()
    gt_labels = dataset.labels.cpu().numpy()
    if dataset.norm:
        gt_embs=  dataset.transback(gt_embs)
        samples= dataset.transback(samples)
    # if local_rank == 0:
    #     save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch // params.clsnum)
    np.save(os.path.join(params.samdir, f'samples_{samples.shape[0]}_diffusion_{params.epoch}_{params.w}.npy'),samples)
    np.save(os.path.join(params.samdir, f'labels_{labels.shape[0]}_diffusion_{params.epoch}_{params.w}.npy'),labels)
    utils.plot_tsne(gt_embs[:50000], gt_labels[:50000], samples[:50000], labels[:50000], params.samdir, 999999)
    #destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1000,help='epochs for loading models')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=20,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=7,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--genum',type=int,default=5600,help='num of generated samples')
    parser.add_argument('--nettype',default="unet_1d",type=str,help='denoiser type')
    parser.add_argument('--inputsize', type=int,default=64, help='1d input size')
    #normalization parameters comes from training data
    parser.add_argument('--norm',type=int,default=1,help='whether normalize data')
    parser.add_argument('--datadir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy',help='data dir')
    parser.add_argument('--labeldir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_gt_labels.npy',help='label dir')
    parser.add_argument('--batchsize',type=int,default=256,help='for dataloader init')
    parser.add_argument('--ema_update_every',type=int,default=10,help='ema update steps')
    parser.add_argument('--ema_decay',type=float,default=0.995,help='ema decay')

    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    main()
