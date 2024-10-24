import os
import copy
import numpy as np
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description='test for diffusion model')
parser.add_argument('--dataset',type=str,default="ogbn-arxiv",choices=['cora', 'citeseer', 'polblogs', 'pubmed', 'cs', 'ogbn-arxiv'], help='dataset name')
parser.add_argument('--batchsize',type=int,default=32786)
parser.add_argument('--gpu_id',type=str, default="0", help='gpu id')
args = parser.parse_args()
root_dir = "/data/yyang409/changyu/generate_graph_embbedding"
model_dir = "/data/yyang409/changyu/expe"


feat_emb_dim = 512
lr_vae = '2e-4'
lr_str = '2.4'
epoch_1_vae = 30000
epoch_2_vae = 5000
epoch_3_vae = 5000

save_interval = 1000

embd_dir = f"{root_dir}/gcl_embeddings/{args.dataset}/all_embs.npy"
label_dir = f"{root_dir}/gcl_embeddings/{args.dataset}/all_ps_labels.npy"
adj_dir = f"{root_dir}/gcl_embeddings/{args.dataset}/adj.npz"
clusterdir = f"{root_dir}/gcl_embeddings/{args.dataset}/all_ps_cluster_kmeans.npy"


vae_dir_1 = f"{model_dir}/gge/large_graphvae_{args.dataset}_only_feat_lr{lr_str}"
vae_dir_2 = f"{model_dir}/gge/large_graphvae_{args.dataset}_freeze_enc_feat1_nmap1_cmap0_lr{lr_str}"
vae_dir_3 = f"{model_dir}/gge/large_graphvae_{args.dataset}_freeze_enc_feat1_nmap1_cmap1_lr{lr_str}"
diffusion_dir = f"{model_dir}/gge/unet_1d_{args.dataset}_64_large_graphvae_encode_all_norm_ema"

labels = np.load(label_dir)
datalen = labels.shape[0]
clsnum = np.unique(labels).shape[0]
genbatch = (3000//clsnum) * clsnum
genum = (datalen//genbatch + 1 ) *  genbatch * 0.6
vae_encode_data = f"{vae_dir_3}/{args.dataset}_latents_{datalen}_largegvae_{epoch_3_vae}_64_encode.npy"
diffusion_sample_data = f"{diffusion_dir}/samples_{genum}_diffusion_3000_1.8.npy"
diffusion_sample_label = f"{diffusion_dir}/labels_{genum}_diffusion_3000_1.8.npy"
diffusion_batch_size = 4096
neighbor_map_dim = 1412
cluster_num = 120
batchsize = args.batchsize


#result dirs
vae_decode_feature = f"{vae_dir_3}/{args.dataset}_latents_{genum}_large_graphvae_{epoch_3_vae}_64_decode_feat.npy"
vae_decode_map = f"{vae_dir_3}/{args.dataset}_latents_{genum}_large_graphvae_{epoch_3_vae}_64_decode_map.npy"

python_path = "/data/yyang409/changyu/graph_diffusion/bin/python"

def command(args):
    script_vae_1 = f'CUDA_VISIBLE_DEVICES={args.gpu_id}  {python_path}  {root_dir}/train_large_graph_vae.py --moddir {vae_dir_1} --samdir {vae_dir_1} --datadir {embd_dir} --labeldir {label_dir}  --adjdir {adj_dir} --clusterdir {clusterdir} --lr {lr_vae} --coef_recon 1 --coef_map 0 --freeze 0 --interval {save_interval} --intervalplot {save_interval} --epoch {epoch_1_vae} --dataset {args.dataset} --neighbor_map_dim {neighbor_map_dim} --cluster_num {cluster_num} --shuffle --batchsize {batchsize}'

    script_vae_2 = f'CUDA_VISIBLE_DEVICES={args.gpu_id}  {python_path}  {root_dir}/train_large_graph_vae.py --moddir {vae_dir_2} --samdir {vae_dir_2} --datadir {embd_dir} --labeldir {label_dir}  --adjdir {adj_dir} --clusterdir {clusterdir} --checkpoint_path {vae_dir_1}/ckpt_{epoch_1_vae}_checkpoint.pt --lr {lr_vae} --coef_recon 1 --coef_map 1 --coef_cmap 0 --coef_nmap 1 --freeze 1 --interval {save_interval} --intervalplot {save_interval} --epoch {epoch_2_vae} --dataset {args.dataset} --neighbor_map_dim {neighbor_map_dim} --cluster_num {cluster_num} --shuffle --batchsize {batchsize}'

    script_vae_3 = f'CUDA_VISIBLE_DEVICES={args.gpu_id}  {python_path}  {root_dir}/train_large_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {embd_dir} --labeldir {label_dir}  --adjdir {adj_dir} --clusterdir {clusterdir} --checkpoint_path {vae_dir_2}/ckpt_{epoch_2_vae}_checkpoint.pt --lr {lr_vae} --coef_recon 1 --coef_map 1 --coef_cmap 1 --coef_nmap 1 --freeze 1 --freeze_cmap 0 --freeze_nmap 1 --freeze_feat 1 --interval {save_interval} --intervalplot {save_interval} --epoch {epoch_3_vae} --dataset {args.dataset} --neighbor_map_dim {neighbor_map_dim} --cluster_num {cluster_num} --shuffle --batchsize {batchsize}'
 
    script_vae_encode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train_large_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {embd_dir} --labeldir {label_dir} --adjdir {adj_dir} --clusterdir {clusterdir} --dataset {args.dataset} --neighbor_map_dim {neighbor_map_dim} --batchsize {batchsize} --run_type encode --lastepo {epoch_3_vae} --feat_emb_dim {feat_emb_dim}'

    script_diffusion = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train.py --batchsize {diffusion_batch_size} --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --interval 500 --intervalplot 500 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datatype gclemb --datadir {vae_encode_data} --labeldir {label_dir} --genum {clsnum*80} --genbatch {clsnum*40} --norm 1'

    script_sample = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path} {root_dir}/sample.py --genum {genum} --genbatch {genbatch}  --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datadir {vae_encode_data} --labeldir {label_dir} --norm 1'

    script_vae_decode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train_large_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {embd_dir} --labeldir {label_dir}  --dataset {args.dataset} --neighbor_map_dim {neighbor_map_dim} --batchsize {batchsize} --run_type decode --lastepo {epoch_3_vae} --datadecode_dir {diffusion_sample_data} --labelsdecode_dir {diffusion_sample_label} --feat_emb_dim {feat_emb_dim}'

    return script_vae_1, script_vae_2, script_vae_3, script_vae_encode, script_diffusion, script_sample, script_vae_decode

def run(args):
    #opt_ = copy.deepcopy(args)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commands = command(args)
    for cmd in commands:
        print(cmd)
        os.system(cmd)
    #output results dir to a file
    expe_setting = f"{args.dataset}"
    with open(f"{model_dir}/gge/commands_large_graphvae.txt", 'a') as command_f:
        command_f.write(timestamp +': \n')
        command_f.write(expe_setting +'\n')
        for cmd in commands:
            command_f.write(cmd+'\n')
        command_f.write(f"############################################################################################## \n")    

    with open(f"{model_dir}/gge/results_large_graphvae.txt", 'a') as result_f:
        result_f.write(timestamp +'\n')
        result_f.write(expe_setting +': \n')
        result_f.write(f"gcl features: \n")
        result_f.write(f"{embd_dir} \n")
        result_f.write(f"---------------------------------------------------------------- \n")
        result_f.write(f"----Synthetic emd, labels, map --------------------------------- \n")
        result_f.write(f"{vae_decode_feature} \n")
        result_f.write(f"{diffusion_sample_label} \n")
        result_f.write(f"{vae_decode_map} \n")

        result_f.write(f"############################################################################################## \n")





    


if __name__ == '__main__':
    
    run(args)