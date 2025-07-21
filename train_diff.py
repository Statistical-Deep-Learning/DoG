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
import dataloader_mnist, dataloader_gclemb
import logging
from math import ceil
import utils
from ema_pytorch import EMA

def train(params:argparse.Namespace):
    if not os.path.exists(params.moddir):
        os.makedirs(params.moddir)
    gfile_stream = open(os.path.join(params.moddir, 'stdout.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info("step: %d, eval_loss: %.5e" % (0, 1))

    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = str(find_free_port())
    #os.environ['WORLD_SIZE'] = '4'
    #os.environ['RANK'] = '0'
    #init_process_group(backend="nccl")
    # get local rank for each process
    # local_rank = get_rank()
    # local_rank = 0
    # set device
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda", 0)
    # load data
    #dataloader, sampler = load_data(params.batchsize, params.numworkers)
    if params.datatype == 'mnist':
        dataloader, _ = dataloader_mnist.load_data(params)
    elif params.datatype == 'gclemb':
        dataloader, dataset = dataloader_gclemb.load_data(params)
    
    # if params.dataname =="cora":
    #     num_labels = 7
    # else:
    #     num_labels = 10
    
    # initialize models
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
    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
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
    if os.path.exists(lastpath):
        ema.load_state_dict(checkpoint["ema"])
    
    # DDP settings 
    # diffusion.model = DDP(
    #                         diffusion.model,
    #                         device_ids = [local_rank],
    #                         output_device = local_rank
    #                     )
    # cemblayer = DDP(
    #                 cemblayer,
    #                 device_ids = [local_rank],
    #                 output_device = local_rank
    #             )
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            # warm_epoch = params.epoch // 10,
                            warm_epoch = params.epoch // 1,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    #cnt = torch.cuda.device_count()
    cnt = 1
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        cemblayer.train()
        total_loss = 0.0
        #sampler.set_epoch(epc)
        # batch iterations
        # with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for img, lab in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                lab = lab.to(device)
                cemb = cemblayer(lab)
                cemb[np.where(np.random.rand(b)<params.threshold)] = 0                       #for unconditional gen
                loss = diffusion.trainloss(x_0, cemb = cemb)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
                total_loss += loss.item()
                ema.update()
        average_loss = total_loss / len(dataloader)
        logging.info("epoch: %d, train_loss: %.5e" % (epc, average_loss))    
        

        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.intervalplot == 0:
            if epc>100 and (epc+1)% params.interval!=0:
                continue
            diffusion.model.eval()
            cemblayer.eval()
            ema.ema_model.eval()
            if params.datatype == 'mnist':
                # generating samples
                # The model generate 80 pictures(8 per row) each time
                # pictures of same row belong to the same class
                all_samples = []
                each_device_batch = params.genbatch // cnt
                with torch.no_grad():
                    lab = torch.ones(params.clsnum, each_device_batch // params.clsnum).type(torch.long) \
                    * torch.arange(start = 0, end = params.clsnum).reshape(-1, 1)
                    lab = lab.reshape(-1, 1).squeeze()
                    lab = lab.to(device)
                    cemb = cemblayer(lab)
                    #genshape = (each_device_batch , 3, 32, 32)
                    genshape = (each_device_batch ,params.inch, params.inputsize)     #only for 1d data
                    if params.ddim:
                        generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
                    else:
                        generated = diffusion.sample(genshape, cemb = cemb)
                    #TODO tansform result to save format for mnist
                    img = dataloader_mnist.transback(generated)
                    #img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 32, 32).contiguous()
                    img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 1, 28, 28).contiguous()
                    # gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                    # all_gather(gathered_samples, img)
                    # all_samples.extend([img for img in gathered_samples])
                    all_samples.extend(img)
                    samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch, 1, 28, 28)
                    # if local_rank == 0:
                    #     save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch // params.clsnum)
                    save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch // params.clsnum)
            elif params.datatype == 'gclemb':           
                # generating samples
                numloop = ceil(params.genum  / params.genbatch)
                all_samples = []
                all_labels = []
                all_samples_ema = []
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
                        generated = ema.ema_model.sample(genshape, cemb = cemb)
                    all_samples.append(generated)
                    all_labels.append(lab)
                    #all_samples_ema.append(generated_ema)
                    
                gen_embs = torch.concat(all_samples, dim = 0)
                #gen_embs_ema = torch.concat(all_samples_ema, dim = 0)
                gen_labels = torch.concat(all_labels, dim = 0)
                gen_embs = gen_embs.squeeze(1).cpu().numpy()
                #gen_embs_ema = gen_embs_ema.squeeze(1).cpu().numpy()
                gen_labels = gen_labels.cpu().numpy()
                # gt_embs = np.load(params.datadir)
                # gt_labels = np.load(params.labeldir)
                gt_embs = dataset.emb.squeeze(1).cpu().numpy()
                gt_labels = dataset.labels.cpu().numpy()
                #transback/unnormalize
                if dataset.norm:
                    gt_embs=  dataset.transback(gt_embs)
                    gen_embs= dataset.transback(gen_embs)
                    #gen_embs_ema= dataset.transback(gen_embs_ema)
                #utils.plot_tsne(gt_embs, gt_labels, gen_embs, gen_labels, params.samdir, epc)
                #utils.plot_tsne(gt_embs, gt_labels, gen_embs_ema, gen_labels, params.samdir, epc, "ema")
                utils.plot_tsne(gt_embs[:4000], gt_labels[:4000], gen_embs[:2000], gen_labels[:2000], params.samdir, epc, "ema")


        if (epc + 1) % params.interval == 0:
            # save checkpoints
            checkpoint = {      'epoch': epc + 1,
                                'net':diffusion.model.state_dict(),
                                'cemblayer':cemblayer.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict(),
                                'ema': ema.state_dict(),
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()
    #destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
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
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=20,help='save epoch interval')
    parser.add_argument('--intervalplot',type=int,default=20,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=70,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--nettype',default="",type=str,help='denoiser type')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, help='mlp hidden layers size')
    parser.add_argument('--res', type=int,default=0, help='mlp whether use skip conn')
    parser.add_argument('--inputsize', type=int,default=64, help='1d input size')
    parser.add_argument('--datatype',type=str,default='gclemb',help='data type')
    # parser.add_argument('--dataname',type=str,default='cora',help='data name')
    parser.add_argument('--datadir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy',help='data dir')
    parser.add_argument('--labeldir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_gt_labels.npy',help='label dir')
    parser.add_argument('--genum',type=int,default=70,help='num of generated samples')
    parser.add_argument('--norm',type=int,default=1,help='whether normalize data')
    parser.add_argument('--ema_update_every',type=int,default=10,help='ema update steps')
    parser.add_argument('--ema_decay',type=float,default=0.995,help='ema decay')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
