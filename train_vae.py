import torch
import torch.nn as nn
from models import VAE

import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from ema_pytorch import EMA
import logging
import dataloader_mnist, dataloader_gclemb
import utils
from torch.utils.data import Dataset, DataLoader

def loss_vae(x, x_hat, mean, log_var):   #reconstruction + KLD(hat, n(0,1))
    KL_loss = - 0.5 * torch.sum(1 + log_var - log_var.exp() - mean.pow(2))
    recon_loss = nn.functional.mse_loss(x_hat, x, 'sum')
    return KL_loss, recon_loss

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
    #output totol loss, kl losss and reconstruction loss
    logging.info("step: %d,train_loss: %.5e, kl_loss: %.5e, recon_loss: %.5e" % (0, 1, 1, 1))
    
    
    device = torch.device("cuda", 0)
    #read data
    if params.datatype == 'mnist':
        dataloader, _ = dataloader_mnist.load_data(params)
    elif params.datatype == 'gclemb':
        dataloader, dataset = dataloader_gclemb.load_data(params)
    #todo load model
    model = VAE(params.hidden_sizes).to(device)

    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        # load checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['vae'])
        lastepc = params.lastepo
    else:
        lastepc = 0

    ema = EMA(model, beta = params.ema_decay, update_every = params.ema_update_every)
    if os.path.exists(checkpoint_path):
        ema.load_state_dict(checkpoint["ema"])

    optimizer = torch.optim.Adam(model.parameters(), params.lr)
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
    #todo train model


    for epc in range(lastepc, params.epoch):
        model.train()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_recon_loss = 0.0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for img, lab in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                x_hat, mean, log_var = model(x_0)
                kl_loss, recon_loss = loss_vae(x_0, x_hat, mean, log_var)
                loss = params.coef_kl * kl_loss + params.coef_recon * recon_loss
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
                total_kl_loss += kl_loss.item()
                total_recon_loss += recon_loss.item()
                ema.update()
        average_loss = total_loss / len(dataloader)
        average_kl_loss = total_kl_loss / len(dataloader)
        average_recon_loss = total_recon_loss / len(dataloader)
        logging.info("epoch: %d, train_loss: %.5e, kl_loss: %.5e, recon_loss: %.5e" % (epc, average_loss, average_kl_loss, average_recon_loss))  

        if (epc + 1) % params.intervalplot == 0:
            if epc>50 and (epc+1)% params.interval!=0:
                continue
            model.eval()   
            all_samples = []
            all_labels = []
            all_gt = []
            if params.datatype == 'mnist':
                 pass
            elif params.datatype == 'gclemb':  
                with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                    for img, lab in tqdmDataLoader:
                        x_0 = img.to(device)
                        with torch.no_grad():
                            x_hat, _, _ = ema.ema_model(x_0)
                        all_samples.append(x_hat)
                        all_labels.append(lab)
                        all_gt.append(x_0)
                        break
                    #all_samples_ema.append(generated_ema)
                    
                gen_embs = torch.concat(all_samples, dim = 0)
                gen_labels = torch.concat(all_labels, dim = 0)
                all_gt = torch.concat(all_gt, dim = 0)
                gen_embs = gen_embs.squeeze(1).cpu().numpy()
                gen_labels = gen_labels.cpu().numpy()
                all_gt = all_gt.squeeze(1).cpu().numpy()
                #gt_embs = dataset.emb.squeeze(1).cpu().numpy()
                #gt_labels = dataset.labels.cpu().numpy()
                #transback/unnormalize
                if dataset.norm:
                    all_gt=  dataset.transback(all_gt)
                    gen_embs= dataset.transback(gen_embs)
                    #gen_embs_ema= dataset.transback(gen_embs_ema)
                #utils.plot_tsne(gt_embs, gt_labels, gen_embs, gen_labels, params.samdir, epc)
                #utils.plot_tsne(gt_embs, gt_labels, gen_embs_ema, gen_labels, params.samdir, epc, "ema")
                utils.plot_tsne(all_gt, gen_labels, gen_embs, gen_labels, params.samdir, epc, "")              
        #todo save model
        if (epc + 1) % params.interval == 0:
            # save checkpoints
            checkpoint = {      'epoch': epc + 1,
                                'vae':model.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'ema': ema.state_dict(),
                            }
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()

def encode(params:argparse.Namespace):
    device = torch.device("cuda", 0)
    #read data
    if params.datatype == 'mnist':
        dataloader, _ = dataloader_mnist.load_data(params)
    elif params.datatype == 'gclemb':
        dataloader, dataset = dataloader_gclemb.load_data(params)
    
    #load model
    model = VAE(params.hidden_sizes).to(device)

    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['vae'])

    ema = EMA(model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    #encode data
    model.eval()   
    all_latents = []
    all_labels = []
    all_gt = []

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            x_0 = img.to(device)
            with torch.no_grad():
                x_latent = ema.ema_model.encode(x_0)
            all_latents.append(x_latent)
            all_labels.append(lab)
            all_gt.append(x_0)
        
    latents = torch.concat(all_latents, dim = 0)
    gen_labels = torch.concat(all_labels, dim = 0)
    all_gt = torch.concat(all_gt, dim = 0)
    latents = latents.squeeze(1).cpu().numpy()
    gen_labels = gen_labels.cpu().numpy()
    all_gt = all_gt.squeeze(1).cpu().numpy()
    #save data
    np.save(os.path.join(params.samdir, f'latents_{latents.shape[0]}_vae_{params.lastepo}_{params.hidden_sizes[-1]}_encode.npy'),latents)


def decode(params:argparse.Namespace):
    device = torch.device("cuda", 0)
    #read data in order to get denomalize parameter
    if params.datatype == 'mnist':
        dataloader, _ = dataloader_mnist.load_data(params)
    elif params.datatype == 'gclemb':
        dataloader, dataset = dataloader_gclemb.load_data(params)
    
    #read data to decode.
    data_decode_np = np.load(params.datadecode_dir)
    data_decode_dataset = dataloader_gclemb.Decode_Dataset(data_decode_np)
    data_decode_loader = DataLoader(data_decode_dataset, batch_size=params.batchsize, shuffle=False)

    #load model
    model = VAE(params.hidden_sizes).to(device)

    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['vae'])

    ema = EMA(model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    #encode data
    model.eval()   
    all_latents = []
    with tqdm(data_decode_loader, dynamic_ncols=True) as tqdmDataLoader:
        for img in tqdmDataLoader:
            x_0 = img.to(device)
            with torch.no_grad():
                x_latent = ema.ema_model.decode(x_0)
            all_latents.append(x_latent)

        
    latents = torch.concat(all_latents, dim = 0)
    latents = latents.squeeze(1).cpu().numpy()
    if dataset.norm:
        latents=  dataset.transback(latents)
    #plot to check code
    #TODO:the labels for generated data should be passed.
    #utils.plot_tsne(dataset.transback(dataset.emb.squeeze(1).cpu().numpy())[:100], dataset.labels.cpu().numpy()[:100], latents[:100], dataset.labels.cpu().numpy()[:100], params.samdir, params.lastepo, "decode")
    #save data
    np.save(os.path.join(params.samdir, f'latents_{latents.shape[0]}_vae_{params.lastepo}_{params.hidden_sizes[0]}_decode.npy'),latents)

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--epoch',type=int,default=3000,help='epochs for training')
    parser.add_argument('--interval',type=int,default=30,help='save epoch interval')
    parser.add_argument('--intervalplot',type=int,default=1,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='/data-drive/backup/changyu/expe/gge/vae_test',help='model addresses')
    parser.add_argument('--samdir',type=str,default='/data-drive/backup/changyu/expe/gge/vae_test',help='sample addresses')
    #parser.add_argument('--genbatch',type=int,default=70,help='batch size for sampling process')
    #parser.add_argument('--clsnum',type=int,default=7,help='num of label classes')
    #parser.add_argument('--inputsize', type=int,default=64, help='1d input size')
    parser.add_argument('--datatype',type=str,default='gclemb',help='data type')
    # parser.add_argument('--dataname',type=str,default='cora',help='data name')
    parser.add_argument('--datadir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy',help='data dir')
    parser.add_argument('--labeldir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy',help='label dir')
    parser.add_argument('--norm',type=int,default=1,help='whether normalize data')
    parser.add_argument('--ema_update_every',type=int,default=10,help='ema update steps')
    parser.add_argument('--ema_decay',type=float,default=0.995,help='ema decay')
    parser.add_argument('--run_type',type=str,default="train",help='select one from train, encode, or decode')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512,256,128,64], help='mlp hidden layers size')
    parser.add_argument('--lastepo',type=int,default=0,help='index of model to load')
    parser.add_argument('--coef_kl',type=float,default=1.0,help='coefficient for kl loss')
    parser.add_argument('--coef_recon',type=float,default=1.0,help='coefficient for recon loss')
    parser.add_argument('--datadecode_dir',type=str,default='',help='data dir for decoding')
    args = parser.parse_args()
    if args.run_type =="train":
        train(args)
    elif args.run_type =="encode":
        encode(args)
    elif args.run_type =="decode":
        decode(args)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()