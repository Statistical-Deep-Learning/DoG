from node_classification.utils import process
import scipy.sparse as sp
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from ema_pytorch import EMA
import logging
import utils
from torch.utils.data import Dataset, DataLoader
from dataloader_graph_emb_gat import load_graph_emb_dataset, load_large_graph_emb_dataset
from dataloader_graph_emb_gat import Decoder_Dataset




class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.6, temp=0.5):
        super(GATLayer, self).__init__()
        assert out_features % heads == 0
        self.out_features = out_features
        self.in_features = in_features
        self.heads = heads
        self.dropout = dropout

        # Weight matrices
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention coefficients
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        '''
        self.a_mask = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_mask.data, gain=1.414)
        '''

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.temp = temp
    def forward(self, x, adj):
        Wh = torch.mm(x, self.W)  # Linear transformation
        bsz = Wh.size(0)

        # Only consider edges that actually exist (i.e., where adj is nonzero)
        edges_id = adj.nonzero(as_tuple=False)

        # Extract features for the source and target nodes of each edge
        Wh1 = Wh[edges_id[:, 0], :]  # Source node features for each edge
        Wh2 = Wh[edges_id[:, 1], :]  # Target node features for each edge
        
        # Concatenate features from source and target nodes
        e_feat = torch.cat([Wh1, Wh2], dim=1)

        # Apply the shared attention mechanism to every edge
        e = self.leakyrelu(torch.matmul(e_feat, self.a).squeeze(1))
        attention = torch.zeros(bsz, bsz).to(x.device)
        attention[edges_id[:, 0], edges_id[:, 1]] = e

        '''
        # Apply mask
        e_mask = self.leakyrelu(torch.matmul(e_feat, self.a_mask).squeeze(1))
        e_mask = torch.sigmoid(e_mask / self.temp)
        mask = torch.zeros(bsz, bsz).to(x.device)
        mask[edges_id[:, 0], edges_id[:, 1]] = e_mask

        attention = attention * mask
        '''
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to node features
        h_prime = torch.matmul(attention, Wh)

        if self.heads > 1:
            # If multiple heads, split and concat
            h_prime = h_prime.view(bsz, self.heads, self.out_features // self.heads)
            h_prime = torch.mean(h_prime, dim=1)

        return h_prime
'''
    def get_mask(self, x, adj, threshold=0.5):
        Wh = torch.mm(x, self.W)  # Linear transformation
        bsz = Wh.size(0)

        # Only consider edges that actually exist (i.e., where adj is nonzero)
        edges_id = adj.nonzero(as_tuple=False)

        # Extract features for the source and target nodes of each edge
        Wh1 = Wh[edges_id[:, 0], :]  # Source node features for each edge
        Wh2 = Wh[edges_id[:, 1], :]  # Target node features for each edge
        
        # Concatenate features from source and target nodes
        e_feat = torch.cat([Wh1, Wh2], dim=1)

        # Apply mask
        e_mask = self.leakyrelu(torch.matmul(e_feat, self.a_mask).squeeze(1))
        e_mask = torch.sigmoid(e_mask / self.temp)
        mask = torch.zeros(bsz, bsz).to(x.device)
        mask[edges_id[:, 0], edges_id[:, 1]] = e_mask
        
        # Convert mask to binary using threshold
        binary_mask = (mask > threshold).float()

        return binary_mask
'''
class node_encoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64):
        super(node_encoder, self).__init__()
        self.feat_encode = nn.Linear(feat_dim, hidden_dim)
        self.neighbor_encode = GATLayer(feat_dim, hidden_dim)  # GAT layer
        self.latent_encode = nn.Linear(hidden_dim*2, reparam_dim)
        self.mean = nn.Linear(reparam_dim, latent_dim)
        self.log_var = nn.Linear(reparam_dim, latent_dim)

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        z = mean + eps * torch.exp(log_var * 0.5)
        return z
    
    def forward(self, x, normalized_adj, pos_emb):
        feat = F.relu(self.feat_encode(x))    #
        neighbor_feat = F.relu(self.neighbor_encode(x + pos_emb, normalized_adj))    #
        feat = torch.cat([feat, neighbor_feat], dim=1)
        feat = F.relu(self.latent_encode(feat))
        mean = self.mean(feat)
        log_var = self.log_var(feat)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

class node_decoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=2708):
        super(node_decoder, self).__init__()
        self.latent_decode = nn.Linear(latent_dim, reparam_dim)
        self.reparam_decode = nn.Linear(reparam_dim, hidden_dim*2)
        self.feat_decode = nn.Linear(hidden_dim, feat_dim)
        self.neighbor_decode = nn.Linear(hidden_dim, neighbor_map_dim)

    def forward(self, z, temp=0.5):
        z_decode_1 = F.relu(self.latent_decode(z))
        z_decode_2 = F.relu(self.reparam_decode(z_decode_1))
        # split z into two parts
        z_decode = torch.chunk(z_decode_2, 2, dim=-1)
        feat = z_decode[0]
        neighbor_feat = z_decode[1]
        feat = self.feat_decode(feat)
        neighbor_feat = self.neighbor_decode(neighbor_feat)
        feat = torch.sigmoid(feat)
        neighbor_map = torch.sigmoid(neighbor_feat/temp)
        # make neighbor_feat sharper
        return feat, neighbor_map

class node_vae(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=2708):
        super(node_vae, self).__init__()
        self.encoder = node_encoder(feat_dim, hidden_dim, reparam_dim, latent_dim)
        self.decoder = node_decoder(feat_dim, hidden_dim, reparam_dim, latent_dim, neighbor_map_dim=neighbor_map_dim)

    def forward(self, feat, normalized_adj, pos_emb):
        z, mean, log_var = self.encoder(feat, normalized_adj, pos_emb)
        feat, neighbor_map = self.decoder(z)
        return feat, neighbor_map, mean, log_var, z

def train(params:argparse.Namespace):
    '''we train the model with the full batch!!!!!!'''
    
    if not os.path.exists(params.moddir):
        os.makedirs(params.moddir)
    gfile_stream = open(os.path.join(params.moddir, 'stdout.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    lastepc = params.lastepo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader, dataset = load_graph_emb_dataset(params)

    vae_model = node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=params.neighbor_map_dim).to(device)
    

    if params.checkpoint_path=="":
        checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    else:
        checkpoint_path = params.checkpoint_path
    if os.path.exists(checkpoint_path):
        # load checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vae_model.load_state_dict(checkpoint['vae'])
        lastepc = params.lastepo
    else:
        lastepc = 0

    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    if os.path.exists(checkpoint_path):
        ema.load_state_dict(checkpoint["ema"])
    
    optimizer = torch.optim.Adam(vae_model.parameters(), params.lr)
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])


    #freeze some layers of the vae_model
    if params.freeze:
        for name, param in vae_model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    for epc in range(lastepc, params.epoch):
        vae_model.train()
        total_loss = 0.0
        total_recon_feature_loss = 0.0
        total_recon_map_loss = 0.0
        #total_edge_loss = 0.0
        #edge_mask = None


    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for feature_emb_gt, normalized_adj, neighbor_map_gt, labels,  positional_embeddings, _ in tqdmDataLoader:
            #We use full batch to train the model
            #if edge_mask is None:
            #    edge_mask = neighbor_map_gt.to(device)
            feature_emb_gt = feature_emb_gt.to(device)
            normalized_adj = normalized_adj.to(device)
            neighbor_map_gt = neighbor_map_gt.to(device)
            labels = labels.to(device)
            positional_embeddings = positional_embeddings.to(device)

            for epc in range(lastepc, params.epoch):
                if epc%1000 == 0:
                    print(f"{epc}th epoch")
                vae_model.train()
                total_loss = 0.0
                total_recon_feature_loss = 0.0
                total_recon_map_loss = 0.0
                total_edge_loss = 0.0
                total_edge_loss_map=0.0

                optimizer.zero_grad()
                reconstructed_feat, neighbor_map, _, _, z = vae_model(feature_emb_gt, normalized_adj, positional_embeddings)
            
                #norm_reconstructed_feat = reconstructed_feat / (reconstructed_feat.norm(p=2, dim=1, keepdim=True)+1e-6)
                bce_loss = F.binary_cross_entropy(neighbor_map, neighbor_map_gt, reduction='mean')
                l2_loss = F.mse_loss(reconstructed_feat, feature_emb_gt, reduction='mean')
                '''
                edge_loss_map = F.binary_cross_entropy(neighbor_map, neighbor_map_gt*edge_mask, reduction='mean')
                edge_loss = F.mse_loss(torch.matmul(reconstructed_feat, reconstructed_feat.T), neighbor_map_gt*edge_mask, reduction='mean')
                '''

                loss = params.coef_recon * l2_loss +  params.coef_map * bce_loss #  + params.factor * edge_loss + params.factor_edgemap * edge_loss_map 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_recon_feature_loss += l2_loss.item()
                total_recon_map_loss += bce_loss.item()
                #total_edge_loss += edge_loss.item()
                #total_edge_loss_map += edge_loss_map.item()
                ema.update()
                
                #with torch.no_grad():
                #    edge_mask = vae_model.encoder.neighbor_encode.get_mask(feature_emb_gt, normalized_adj)
    
                average_loss = total_loss / len(dataloader)
                average_recon_feature_loss = total_recon_feature_loss / len(dataloader)
                average_recon_map_loss = total_recon_map_loss / len(dataloader)
                #average_edge_loss = total_edge_loss / len(dataloader)
                #average_edge_loss_map = total_edge_loss_map / len(dataloader)
                #logging.info("epoch: %d,  recon_feat_loss: %.5e, edge_loss: %.5e,   edge_loss_map: %.5e, recon_map_loss: %.5e, train_loss: %.5e," % (epc,  average_recon_feature_loss, average_edge_loss, average_edge_loss_map, average_recon_map_loss, average_loss,))  
                logging.info("epoch: %d,  recon_feat_loss: %.5e, recon_map_loss: %.5e, train_loss: %.5e," % (epc,  average_recon_feature_loss, average_recon_map_loss, average_loss,))  
        
                if (epc + 1) % params.interval == 0:
                # save checkpoints
                    checkpoint = {      'epoch': epc + 1,
                                'vae':vae_model.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'ema': ema.state_dict(),
                            }
                    torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))

                torch.cuda.empty_cache()
                if (epc + 1) % params.intervalplot == 0:
                    if epc>50 and (epc+1)% params.interval!=0:
                        continue
                    ema.ema_model.eval()   
                    all_samples = []
                    all_labels = []
                    all_gt = []
                    with torch.no_grad():
                        reconstructed_feat, neighbor_map, _, _, _ = ema.ema_model(feature_emb_gt, normalized_adj, positional_embeddings)
                    all_samples.append(reconstructed_feat)
                    all_labels.append(labels)
                    all_gt.append(feature_emb_gt)
                    
                    gen_embs = torch.concat(all_samples, dim = 0)
                    gen_labels = torch.concat(all_labels, dim = 0)
                    all_gt = torch.concat(all_gt, dim = 0)
                    gen_embs = gen_embs.squeeze(1).cpu().numpy()
                    gen_labels = gen_labels.cpu().numpy()
                    all_gt = all_gt.squeeze(1).cpu().numpy()
                    utils.plot_tsne(all_gt[:1000], gen_labels[:1000], gen_embs[:1000], gen_labels[:1000], params.samdir, epc, f"map_{params.coef_map}")


def encode(params:argparse.Namespace):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader, dataset = load_graph_emb_dataset(params)
    vae_model = node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=params.neighbor_map_dim).to(device)
    
    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vae_model.load_state_dict(checkpoint['vae'])
    
    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    vae_model.eval()   
    all_latents = []

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for feature_emb_gt, normalized_adj, neighbor_map_gt, labels,  positional_embeddings, _ in tqdmDataLoader:  
            feature_emb_gt = feature_emb_gt.to(device)
            normalized_adj = normalized_adj.to(device)
            neighbor_map_gt = neighbor_map_gt.to(device)
            positional_embeddings = positional_embeddings.to(device)
            with torch.no_grad():
                z, mean, log_var = ema.ema_model.encoder(feature_emb_gt, normalized_adj, positional_embeddings)
            all_latents.append(mean)
            #all_labels.append(labels)
    latents = torch.concat(all_latents, dim = 0)
    latents = latents.squeeze(1).cpu().numpy()
    #save data
    np.save(os.path.join(params.samdir, f'{params.dataset}_latents_{latents.shape[0]}_gvae_{params.lastepo}_{latents.shape[-1]}_encode.npy'),latents)



    
def decode(params:argparse.Namespace):
    #don't shuffle data when testing
    labels_decode = np.load(params.labelsdecode_dir)
    embeds = np.load(params.datadir)
    labels = np.load(params.labeldir)
    _, dataset = load_graph_emb_dataset(params)
    decode_dataset = Decoder_Dataset(params.datadecode_dir)
    decode_dataloader = DataLoader(decode_dataset, batch_size=params.batchsize, shuffle=params.shuffle, num_workers=params.numworkers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    vae_model = node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=params.neighbor_map_dim).to(device)
    
    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vae_model.load_state_dict(checkpoint['vae'])
    
    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    ema.ema_model.eval()   
    all_feats = []
    all_neighbor_maps = []
    with tqdm(decode_dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for data_decode in tqdmDataLoader:
            data_decode = data_decode.to(device)
            with torch.no_grad():
                feat, neighbor_map = ema.ema_model.decoder(data_decode)
            all_feats.append(feat.cpu())
            all_neighbor_maps.append(neighbor_map.cpu())
            
    allfeats = torch.concat(all_feats, dim = 0)
    allneighbor_maps = torch.concat(all_neighbor_maps, dim = 0)

    allfeats = allfeats.squeeze(1).cpu().numpy()
    allneighbor_maps = allneighbor_maps.squeeze(1).cpu().numpy()
    if params.norm: #denormalize data
        origianal_feature_embs = np.load(params.datadir)
        data_min = origianal_feature_embs.min()
        data_max = origianal_feature_embs.max()
        allfeats=  allfeats * (data_max.item() - data_min.item()) + data_min.item()
    #plot to check code: when test
    utils.plot_tsne(embeds[:3000], labels[:3000], allfeats[:3000], labels_decode[:3000], params.samdir, params.lastepo, "decode")
    #save data
    np.save(os.path.join(params.samdir, f'{params.dataset}_latents_{allfeats.shape[0]}_gvae_{params.lastepo}_{data_decode.shape[-1]}_decode_feat.npy'),allfeats)
    np.save(os.path.join(params.samdir, f'{params.dataset}_latents_{allneighbor_maps.shape[0]}_gvae_{params.lastepo}_{data_decode.shape[-1]}_decode_map.npy'),allneighbor_maps)

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
    parser.add_argument('--moddir',type=str,default='/data-drive/backup/changyu/expe/gge/graphvae_test',help='model addresses')
    parser.add_argument('--samdir',type=str,default='/data-drive/backup/changyu/expe/gge/graphvae_test',help='sample addresses')
    #parser.add_argument('--genbatch',type=int,default=70,help='batch size for sampling process')
    #parser.add_argument('--clsnum',type=int,default=7,help='num of label classes')
    #parser.add_argument('--inputsize', type=int,default=64, help='1d input size')
    parser.add_argument('--datatype',type=str,default='gclemb',help='data type')
    # parser.add_argument('--dataname',type=str,default='cora',help='data name')
    parser.add_argument('--datadir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy',help='data dir')
    parser.add_argument('--labeldir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy',help='label dir')
    parser.add_argument('--adjdir',type=str,default='/',help='adj dir')
    parser.add_argument('--norm',type=int,default=1,help='whether normalize data')
    parser.add_argument('--ema_update_every',type=int,default=10,help='ema update steps')
    parser.add_argument('--ema_decay',type=float,default=0.995,help='ema decay')
    parser.add_argument('--run_type',type=str,default="train",help='select one from train, encode, or decode')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512,256,128,64], help='mlp hidden layers size')
    parser.add_argument('--lastepo',type=int,default=0,help='index of model to load')
    parser.add_argument('--coef_kl',type=float,default=1.0,help='coefficient for kl loss')
    parser.add_argument('--coef_recon',type=float,default=1.0,help='coefficient for recon features loss')
    parser.add_argument('--coef_map',type=float,default=1.0,help='coefficient for recon map loss')
    parser.add_argument('--datadecode_dir',type=str,default='',help='data dir for decoding')
    parser.add_argument('--labelsdecode_dir',type=str,default='',help='labels dir for decoding data')
    parser.add_argument('--checkpoint_path',type=str,default='',help='checkpoint_path')
    parser.add_argument('--freeze',type=int,default=0,help='whether freeze encoder layers')
    parser.add_argument('--neighbor_map_dim',type=int,default=2708, help='neighbor map dimension,number of dataset nodes')    
    parser.add_argument('--feat_emb_dim',type=int,default=512,help='feature embedding dimension')
    parser.add_argument('--shuffle', action='store_true', help="If present, shuffle data.")
    parser.add_argument('--dataset',type=str,default="cora",help='dataset name')
    #parser.add_argument('--factor',type=float,default=0.00001,help='coefficient for edge loss')
    #parser.add_argument('--factor_edgemap',type=float,default=0,help='coefficient for recon edge_map loss')
    

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