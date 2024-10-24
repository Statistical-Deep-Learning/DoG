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
from dataloader_graph_emb import load_large_graph_emb_dataset, load_large_graph_emb_dataset_for_encoding
from dataloader_graph_emb import Decoder_Dataset

class node_encoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64):
        super(node_encoder, self).__init__()
        self.feat_encode = nn.Linear(feat_dim, hidden_dim)
        self.neighbor_encode = nn.Linear(feat_dim, hidden_dim)
        self.latent_encode = nn.Linear(hidden_dim*2, reparam_dim)
        self.mean = nn.Linear(reparam_dim, latent_dim)
        self.log_var = nn.Linear(reparam_dim, latent_dim)

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        z = mean + eps * torch.exp(log_var * 0.5)
        return z
    
    def forward(self, feat, neighbor_feat):
        #for name, param in self.named_parameters():
        #    print(f"name: {name},max:{param.max()},min{param.min()}:.mean:{param.mean()}")
        feat = F.relu(self.feat_encode(feat))
        neighbor_feat = F.relu(self.neighbor_encode(neighbor_feat))
        feat = torch.cat([feat, neighbor_feat], dim=1)
        feat = F.relu(self.latent_encode(feat))
        mean = self.mean(feat)
        log_var = self.log_var(feat)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

class node_decoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, cluster_embed_dim=128, neighbor_map_dim=101, cluster_num=27, temp=0.5, index_dict=None):
        super(node_decoder, self).__init__()
        self.latent_decode = nn.Linear(latent_dim, reparam_dim)
        self.reparam_decode = nn.Linear(reparam_dim, hidden_dim*3)
        self.feat_decode = nn.Linear(hidden_dim, feat_dim)
        self.neighbor_decode = nn.Linear(hidden_dim, hidden_dim*2)
        self.neighbor_decode_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.neighbor_decode_2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.neighbor_decode_3 = nn.Linear(hidden_dim*2, neighbor_map_dim)
        self.cluster_to_neighbor = nn.Linear(cluster_embed_dim, hidden_dim)
        self.cluster_ind_embedding = nn.Embedding(cluster_num, cluster_embed_dim)


        self.cluster_decode_1 = nn.Linear(hidden_dim, hidden_dim)
        self.cluster_decode_2 = nn.Linear(hidden_dim, hidden_dim)
        self.cluster_decode_3 = nn.Linear(hidden_dim, hidden_dim)
        self.cluster_decode = nn.Linear(hidden_dim, cluster_num)

        self.temp = temp
        self.index_dict = torch.tensor(index_dict)


        

    def forward(self, z, cluster_ind_of_neighbors=None):
        self.index_dict = self.index_dict.to(z.device)
        z_decode_1 = F.relu(self.latent_decode(z))
        z_decode_2 = F.relu(self.reparam_decode(z_decode_1))
        # split z into two parts
        z_decode = torch.chunk(z_decode_2, 3, dim=-1)
        feat = z_decode[0]
        neighbor_feat = z_decode[1]
        cluster = z_decode[2]

        feat = self.feat_decode(feat)
        feat = torch.sigmoid(feat)
        cluster = F.relu(self.cluster_decode_1(cluster))
        cluster = F.relu(self.cluster_decode_2(cluster))
        cluster = F.relu(self.cluster_decode_3(cluster))
        cluster = self.cluster_decode(cluster)
        cluster = torch.sigmoid(cluster/self.temp)

        if self.training:
            cluster_ind_embedding = self.cluster_ind_embedding(cluster_ind_of_neighbors.long())
            cluster_to_neighbor = F.relu(self.cluster_to_neighbor(cluster_ind_embedding))
            neighbor_feat = self.neighbor_decode(neighbor_feat + cluster_to_neighbor)

            neighbor_feat = F.relu(neighbor_feat)
            neighbor_feat = F.relu(self.neighbor_decode_1(neighbor_feat))
            neighbor_feat = F.relu(self.neighbor_decode_2(neighbor_feat))
            neighbor_feat = self.neighbor_decode_3(neighbor_feat)

            neighbor_map = torch.sigmoid(neighbor_feat/self.temp)
            return feat, neighbor_map, cluster, 1
        else:
            cluster_ind_map = cluster >0.9
            cluster_indexes_2d = torch.nonzero(cluster_ind_map)
            #neighbor_map_ingraph_all = torch.zeros((0, len(self.index_dict)))
            neighbor_map_ingraph_all=[]
            for i in range(cluster_ind_map.shape[0]):
                cluster_indexes = cluster_indexes_2d[cluster_indexes_2d[:,0]==i, 1]
                index_ingraph_all = torch.tensor([]).to(z.device)
                for cluster_ind_of_neighbors in cluster_indexes:
                    cluster_ind_embedding = self.cluster_ind_embedding(cluster_ind_of_neighbors)
                    cluster_to_neighbor = F.relu(self.cluster_to_neighbor(cluster_ind_embedding))
                    neighbor_feat_single = self.neighbor_decode(neighbor_feat[i] + cluster_to_neighbor)
                    neighbor_feat_single = F.relu(neighbor_feat_single)
                    neighbor_feat_single = F.relu(self.neighbor_decode_1(neighbor_feat_single))
                    neighbor_feat_single = F.relu(self.neighbor_decode_2(neighbor_feat_single))
                    neighbor_feat_single = self.neighbor_decode_3(neighbor_feat_single)
                    neighbor_map = torch.sigmoid(neighbor_feat_single/self.temp)
                    neighbor_map = neighbor_map>0.9
                    #transform back to the original index in the graph
                    neighbor_ind_incluster = torch.nonzero(neighbor_map)[:,0]
                    cluster_cond = self.index_dict[:,1]== cluster_ind_of_neighbors
                    neighbor_cond = torch.isin(self.index_dict[:,2], neighbor_ind_incluster)
                    index_ingraph = self.index_dict[cluster_cond & neighbor_cond][:, 0]
                    index_ingraph_all = torch.cat((index_ingraph_all, index_ingraph), dim=0)
                neighbor_map_ingraph = torch.zeros((len(self.index_dict)))
                neighbor_map_ingraph[index_ingraph_all.long()] = 1
                #neighbor_map_ingraph_all = torch.cat((neighbor_map_ingraph_all, neighbor_map_ingraph.reshape(1, -1)), dim=0)
                neighbor_map_ingraph_all.append(neighbor_map_ingraph.reshape(1, -1))
            neighbor_map_ingraph_all=torch.stack(neighbor_map_ingraph_all).squeeze()
            return feat, 1, cluster, neighbor_map_ingraph_all

class hier_node_vae(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, cluster_embed_dim=128, cluster_num=27, neighbor_map_dim=101,index_dict=None):
        super(hier_node_vae, self).__init__()
        self.encoder = node_encoder(feat_dim, hidden_dim, reparam_dim, latent_dim)
        self.decoder = node_decoder(feat_dim, hidden_dim, reparam_dim, latent_dim, cluster_embed_dim =cluster_embed_dim, 
                                     cluster_num= cluster_num, neighbor_map_dim= neighbor_map_dim, index_dict=index_dict)

    def forward(self, feat, neighbor_feat, cluster_ind_of_neighbors):
        z, mean, log_var = self.encoder(feat, neighbor_feat)
        feat, neighbor_map, cluster_map, neighbor_map_ingraph = self.decoder(z, cluster_ind_of_neighbors)
        return feat, neighbor_map, cluster_map, neighbor_map_ingraph, mean, log_var


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lastepc = params.lastepo

    dataloader, index_dict = load_large_graph_emb_dataset(params)
    
    vae_model = hier_node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, cluster_embed_dim =             
                 params.cluster_embed_dim, cluster_num=params.cluster_num, neighbor_map_dim=params.neighbor_map_dim, index_dict=index_dict).to(device)


    if params.checkpoint_path=="":
        checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    else:
        checkpoint_path = params.checkpoint_path
    if os.path.exists(checkpoint_path):
        # load checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vae_model.load_state_dict(checkpoint['vae'], strict=False)
        lastepc = params.lastepo
    else:
        lastepc = 0

    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    if os.path.exists(checkpoint_path):
        ema.load_state_dict(checkpoint["ema"], strict=False)
    
    optimizer = torch.optim.Adam(vae_model.parameters(), params.lr)
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])


    #freeze some layers of the vae_model
    if params.freeze:  #for encoder
        for name, param in vae_model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False    
    
    if params.freeze_cmap:
        for name, param in vae_model.named_parameters():
            if "cluster_decode" in name: 
                param.requires_grad = False
            #print(f"name: {name}, requires_grad: {param.requires_grad}")
    if params.freeze_nmap:
        for name, param in vae_model.named_parameters():
            if "neighbor_decode" in name or "cluster_ind_embedding" in name or "cluster_to_neighbor" in name:
                param.requires_grad = False
            #print(f"name: {name}, requires_grad: {param.requires_grad}")
    if params.freeze_feat:
        for name, param in vae_model.named_parameters():
            if "feat_decode" in name or "latent_decode" in name or "reparam_decode" in name:
                param.requires_grad = False
            #print(f"name: {name}, requires_grad: {param.requires_grad}")
    if params.freeze_public_layers:
        for name, param in vae_model.named_parameters():
            if "latent_decode" in name or "reparam_decode" in name:
                param.requires_grad = False
            #print(f"name: {name}, requires_grad: {param.requires_grad}")

    for epc in range(lastepc, params.epoch):
        vae_model.train()
        total_loss = 0.0
        total_recon_feature_loss = 0.0
        total_neighbor_map_loss = 0.0
        total_cluster_map_loss = 0.0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for feature_emb_gt, neighbor_feats_emb, neighbor_map_incluster_gt, cluster_map_gt, cluster_ind_of_neighbors, label in tqdmDataLoader:
                optimizer.zero_grad()
                feature_emb_gt = feature_emb_gt.to(device)
                neighbor_feats_emb = neighbor_feats_emb.to(device)
                neighbor_map_incluster_gt = neighbor_map_incluster_gt.to(device)
                cluster_map_gt = cluster_map_gt.to(device)
                cluster_ind_of_neighbors = cluster_ind_of_neighbors.to(device)

                reconstructed_feat, neighbor_map, cluster_map, _, _, _ = vae_model(feature_emb_gt, neighbor_feats_emb, cluster_ind_of_neighbors)
		        
                # binary cross entropy loss between reconstructed neighbor map and ground truth neighbor map
                nmap_bce_loss = F.binary_cross_entropy(neighbor_map, neighbor_map_incluster_gt, reduction='mean')
                cmap_bce_loss = F.binary_cross_entropy(cluster_map, cluster_map_gt, reduction='mean')        
                # l2 loss between reconstructed node features and ground truth node features
                l2_loss = F.mse_loss(reconstructed_feat, feature_emb_gt, reduction='mean')
		        
                loss = params.coef_recon * l2_loss + params.coef_map * (params.coef_nmap * nmap_bce_loss + params.coef_cmap * cmap_bce_loss) 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_recon_feature_loss += l2_loss.item()
                total_neighbor_map_loss += nmap_bce_loss.item()
                total_cluster_map_loss += cmap_bce_loss.item()
                ema.update()

            average_loss = total_loss / len(dataloader)
            average_recon_feature_loss = total_recon_feature_loss / len(dataloader)
            average_neighbor_map_loss = total_neighbor_map_loss / len(dataloader)
            average_cluster_map_loss = total_cluster_map_loss / len(dataloader)
            logging.info("epoch: %d, train_loss: %.5e, recon_feat_loss: %.5e, nmap_loss: %.5e, , cmap_loss: %.5e" % (epc, average_loss, average_recon_feature_loss, average_neighbor_map_loss, average_cluster_map_loss))
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
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for feature_emb_gt, neighbor_feats_emb, neighbor_map_incluster_gt, cluster_map_gt, cluster_ind_of_neighbors, label in tqdmDataLoader:
                    feature_emb_gt = feature_emb_gt.to(device)
                    neighbor_feats_emb = neighbor_feats_emb.to(device)
                    neighbor_map_incluster_gt = neighbor_map_incluster_gt.to(device)
                    cluster_map_gt = cluster_map_gt.to(device)
                    cluster_ind_of_neighbors = cluster_ind_of_neighbors.to(device)
                    with torch.no_grad():
                        reconstructed_feat, _, cluster_map, neighbor_map_ingraph,  _, _ = ema.ema_model(feature_emb_gt, neighbor_feats_emb, cluster_ind_of_neighbors)
                    all_samples.append(reconstructed_feat)
                    all_labels.append(label)
                    all_gt.append(feature_emb_gt)
                    
            gen_embs = torch.concat(all_samples, dim = 0)
            gen_labels = torch.concat(all_labels, dim = 0)
            all_gt = torch.concat(all_gt, dim = 0)
            gen_embs = gen_embs.squeeze(1).cpu().numpy()
            gen_labels = gen_labels.cpu().numpy()
            all_gt = all_gt.squeeze(1).cpu().numpy()
            utils.plot_tsne(all_gt[:2000], gen_labels[:2000], gen_embs[:2000], gen_labels[:2000], params.samdir, epc, f"map_{params.coef_map}")



def encode(params:argparse.Namespace):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader, dataset = load_large_graph_emb_dataset_for_encoding(params)
    _, index_dict = load_large_graph_emb_dataset(params)
    vae_model = hier_node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, cluster_embed_dim = params.cluster_embed_dim, cluster_num=params.cluster_num, neighbor_map_dim=params.neighbor_map_dim, index_dict= index_dict).to(device)
    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vae_model.load_state_dict(checkpoint['vae'])
    
    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])

    ema.ema_model.eval()   
    all_latents = []
    #all_labels.append(labels)
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for feature_emb_gt, neighbor_feats_emb, labels in tqdmDataLoader:
            feature_emb_gt = feature_emb_gt.to(device)
            neighbor_feats_emb = neighbor_feats_emb.to(device)
            with torch.no_grad():
                z, mean, log_var = ema.ema_model.encoder(feature_emb_gt, neighbor_feats_emb)
            all_latents.append(mean)

            
    latents = torch.concat(all_latents, dim = 0)
    latents = latents.squeeze(1).cpu().numpy()
    #save data
    np.save(os.path.join(params.samdir, f'{params.dataset}_latents_{latents.shape[0]}_largegvae_{params.lastepo}_{latents.shape[-1]}_encode.npy'),latents)


def decode(params:argparse.Namespace):
    #read original data for index_dict
    _, dataset = load_large_graph_emb_dataset_for_encoding(params)
    _, index_dict = load_large_graph_emb_dataset(params) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae_model = hier_node_vae(feat_dim=params.feat_emb_dim, hidden_dim=256, reparam_dim=128, latent_dim=64, cluster_embed_dim = params.cluster_embed_dim, cluster_num=params.cluster_num, neighbor_map_dim=params.neighbor_map_dim, index_dict=index_dict).to(device)
    
    
    checkpoint_path = os.path.join(params.moddir, f'ckpt_{params.lastepo}_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vae_model.load_state_dict(checkpoint['vae'])
    
    ema = EMA(vae_model, beta = params.ema_decay, update_every = params.ema_update_every)
    ema.load_state_dict(checkpoint["ema"])
    ema.ema_model.eval()  

    decode_dataset = Decoder_Dataset(params.datadecode_dir)
    decode_dataloader = DataLoader(decode_dataset, batch_size=params.batchsize, shuffle=params.shuffle, num_workers=params.numworkers)

    all_feats = []
    all_neighbor_maps = []
    with tqdm(decode_dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for data_decode in tqdmDataLoader:
            data_decode = data_decode.to(device)
            with torch.no_grad():
                feat, _, _, neighbor_map_ingraph = ema.ema_model.decoder(data_decode)
            all_feats.append(feat)
            neighbor_map_ingraph = sp.csr_matrix(neighbor_map_ingraph.cpu().numpy())
            all_neighbor_maps.append(neighbor_map_ingraph)
            
    allfeats = torch.concat(all_feats, dim = 0)
    allneighbor_maps = sp.vstack(all_neighbor_maps, format='csr')

    allfeats = allfeats.squeeze(1).cpu().numpy()
    if params.norm: #denormalize data
        origianal_feature_embs = np.load(params.datadir)
        data_min = origianal_feature_embs.min()
        data_max = origianal_feature_embs.max()
        allfeats=  allfeats * (data_max.item() - data_min.item()) + data_min.item()
    #plot to check code
    #utils.plot_tsne(embeds[:300], labels[:300], allfeats[:300], labels_decode[:300], params.samdir, params.lastepo, "decode")
    #save data
    np.save(os.path.join(params.samdir, f'{params.dataset}_latents_{allfeats.shape[0]}_lgvae_{params.lastepo}_{data_decode.shape[-1]}_decode_feat.npy'),allfeats)
    #save the map as csr_matrix
    sp.save_npz(os.path.join(params.samdir, f'{params.dataset}_latents_{allneighbor_maps.shape[0]}_lgvae_{params.lastepo}_{data_decode.shape[-1]}_decode_map.npz'),allneighbor_maps)

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
    parser.add_argument('--moddir',type=str,default='/data-drive/backup/changyu/expe/gge/large_graphvae_test',help='model addresses')
    parser.add_argument('--samdir',type=str,default='/data-drive/backup/changyu/expe/gge/large_graphvae_test',help='sample addresses')
    #parser.add_argument('--genbatch',type=int,default=70,help='batch size for sampling process')
    #parser.add_argument('--clsnum',type=int,default=7,help='num of label classes')
    #parser.add_argument('--inputsize', type=int,default=64, help='1d input size')
    parser.add_argument('--datatype',type=str,default='gclemb',help='data type')
    # parser.add_argument('--dataname',type=str,default='cora',help='data name')
    parser.add_argument('--datadir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/ogbn-arxiv/all_embs.npy',help='data dir')
    parser.add_argument('--labeldir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/ogbn-arxiv/all_ps_labels.npy',help='label dir')
    parser.add_argument('--adjdir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/ogbn-arxiv/adj.npz',help='adj dir')
    parser.add_argument('--norm',type=int,default=1,help='whether normalize data')
    parser.add_argument('--ema_update_every',type=int,default=10,help='ema update steps')
    parser.add_argument('--ema_decay',type=float,default=0.995,help='ema decay')
    parser.add_argument('--run_type',type=str,default="train",help='select one from train, encode, or decode')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512,256,128,64], help='mlp hidden layers size')
    parser.add_argument('--lastepo',type=int,default=0,help='index of model to load')
    parser.add_argument('--coef_kl',type=float,default=1.0,help='coefficient for kl loss')
    parser.add_argument('--coef_recon',type=float,default=1.0,help='coefficient for recon features loss')
    parser.add_argument('--coef_map',type=float,default=1.0,help='coefficient for recon map loss')
    parser.add_argument('--coef_cmap',type=float,default=1.0,help='coefficient for cluster map loss')
    parser.add_argument('--coef_nmap',type=float,default=1.0,help='coefficient for nighbor map loss')
    parser.add_argument('--datadecode_dir',type=str,default='',help='data dir for decoding')
    parser.add_argument('--labelsdecode_dir',type=str,default='',help='labels dir for decoding data')
    parser.add_argument('--checkpoint_path',type=str,default='',help='checkpoint_path')
    parser.add_argument('--freeze',type=int,default=0,help='whether freeze encoder layers')
    parser.add_argument('--freeze_cmap',type=int,default=0,help='whether freeze cmap related layers')
    parser.add_argument('--freeze_nmap',type=int,default=0,help='whether freeze nmap related layers')
    parser.add_argument('--freeze_feat',type=int,default=0,help='whether freeze feature renconstruction related layers')
    parser.add_argument('--freeze_public_layers',type=int,default=0,help='whether freeze public layers in decoder')
    parser.add_argument('--neighbor_map_dim',type=int,default=1412, help='max num of nodes in an assigned cluster')    
    parser.add_argument('--clusterdir',type=str,default='/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/ogbn-arxiv/all_ps_cluster_kmeans.npy',help='assigned_cluster dir')
    parser.add_argument('--cluster_num',type=int,default=120,help='number of assigned clusters')
    parser.add_argument('--feat_emb_dim',type=int,default=512,help='feature embedding dimension')
    parser.add_argument('--cluster_embed_dim',type=int,default=128,help='cluster embedding dimension')
    parser.add_argument('--shuffle', action='store_true', help="If present, shuffle data.")
    parser.add_argument('--dataset',type=str,default="ogbn-arxiv",help='dataset name')

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


