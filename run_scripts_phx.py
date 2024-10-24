import os
import copy
import numpy as np
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description='test for diffusion model')
parser.add_argument('--dataset',type=str,default="cora",choices=['cora', 'citeseer', 'polblogs', 'pubmed', 'cs', 'ogbn-arxiv'], help='dataset name')
#parser.add_argument('--att_type',type=str,default="meta",choices=['meta', 'nettack', 'random','noattk', 'pgd'], help='attack type')
#parser.add_argument('--att_rate',type=str,default="0.05",choices=['0.05', '0.1', '0.15', '0.2', '0.25','1.0', '2.0', '3.0','4.0', '5.0', '0.0'], help='attack rate')
#parser.add_argument('--edgeloss_type',type=str,default="featmap",choices=['feat', 'map', 'featmap'], help='edge type')
parser.add_argument('--gpu_id',type=str, default="0", help='gpu id')
args = parser.parse_args()
root_dir = "/data/yyang409/changyu/generate_graph_embbedding"
model_dir = "/data/yyang409/changyu/expe"

if args.dataset == 'pubmed':
    feat_emb_dim = 512
    lr_vae = '1e-3'
    lr_str = '1.3'
    epoch_1_vae = 20000
    epoch_2_vae = 10000
elif args.dataset == 'cs':
    feat_emb_dim = 512
    #lr_vae = '1e-3'
    #lr_str = '1.3'
    lr_vae = '2e-4'
    lr_str = '2.4'
    epoch_1_vae = 50000
    epoch_2_vae = 50000
else:
    feat_emb_dim = 512
    lr_vae = '2e-4'
    lr_str = '2.4'
    epoch_1_vae = 50000
    epoch_2_vae = 50000

save_interval = 2000

embd_dir = f"{root_dir}/gcl_embeddings/{args.dataset}/all_embs.npy"
label_dir = f"{root_dir}/gcl_embeddings/{args.dataset}/all_ps_labels.npy"
#adj_dir = f"{model_dir}/graphattk/{args.dataset}_{args.att_type}_adj_{args.att_rate}.npz"
vae_dir_1 = f"{model_dir}/gge/graphvae_gat_{args.dataset}_only_feat_lr{lr_str}"
vae_dir_2 = f"{model_dir}/gge/graphvae_gat_{args.dataset}_freeze_enc_feat_map_lr{lr_str}"

diffusion_dir = f"{model_dir}/gge/unet_1d_{args.dataset}_64_graphvae_gat_encode_all_norm_ema"

labels = np.load(label_dir)
datalen = labels.shape[0]
clsnum = np.unique(labels).shape[0]
genbatch = (3000//clsnum) * clsnum
genum = (datalen//genbatch + 1 ) *  genbatch * 3
vae_encode_data = f"{vae_dir_2}/{args.dataset}_latents_{datalen}_gvae_{epoch_2_vae}_64_encode.npy"
diffusion_sample_data = f"{diffusion_dir}/samples_{genum}_diffusion_3000_1.8.npy"
diffusion_sample_label = f"{diffusion_dir}/labels_{genum}_diffusion_3000_1.8.npy"
diffusion_batch_size = 2048

#result dirs
vae_decode_feature = f"{vae_dir_2}/{args.dataset}_latents_{genum}_graphvae_gat_{epoch_2_vae}_64_decode_feat.npy"
vae_decode_map = f"{vae_dir_2}/{args.dataset}_latents_{genum}_graphvae_gat_{epoch_2_vae}_64_decode_map.npy"

python_path = "/data/yyang409/changyu/graph_diffusion/bin/python"

def command(args):

    script_vae_1 = f'CUDA_VISIBLE_DEVICES={args.gpu_id}  {python_path}  {root_dir}/train_graph_vae_gat.py --moddir {vae_dir_1} --samdir {vae_dir_1} --datadir {embd_dir} --labeldir {label_dir}  --norm 1 --lr {lr_vae} --coef_recon 1 --coef_map 0  --interval {save_interval} --intervalplot 10 --epoch {epoch_1_vae} --freeze 0 --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --feat_emb_dim {feat_emb_dim} --lastepo 20000'

    script_vae_2 = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train_graph_vae_gat.py --moddir {vae_dir_2} --samdir {vae_dir_2} --datadir {embd_dir} --labeldir {label_dir}  --checkpoint_path {vae_dir_1}/ckpt_{epoch_1_vae}_checkpoint.pt --norm 1 --lr {lr_vae} --coef_recon 1 --coef_map 1 --interval {save_interval} --intervalplot 10 --epoch {epoch_2_vae} --freeze 1 --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --feat_emb_dim {feat_emb_dim}'
 
    script_vae_encode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train_graph_vae_gat.py --moddir {vae_dir_2} --samdir {vae_dir_2} --datadir {embd_dir} --labeldir {label_dir}  --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --run_type encode --lastepo {epoch_2_vae} --feat_emb_dim {feat_emb_dim}'

    script_diffusion = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train.py --batchsize {diffusion_batch_size} --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --interval 500 --intervalplot 500 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datatype gclemb --datadir {vae_encode_data} --labeldir {label_dir} --genum {clsnum*80} --genbatch {clsnum*40} --norm 1'

    script_sample = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path} {root_dir}/sample.py --genum {genum} --genbatch {genbatch}  --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datadir {vae_encode_data} --labeldir {label_dir} --norm 1'

    script_vae_decode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {root_dir}/train_graph_vae_gat.py --moddir {vae_dir_2} --samdir {vae_dir_2} --datadir {embd_dir} --labeldir {label_dir}  --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --run_type decode --lastepo {epoch_2_vae} --datadecode_dir {diffusion_sample_data} --labelsdecode_dir {diffusion_sample_label} --feat_emb_dim {feat_emb_dim}'

    return script_vae_1, script_vae_2, script_vae_encode, script_diffusion, script_sample, script_vae_decode

def run(args):
    #opt_ = copy.deepcopy(args)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commands = command(args)
    for cmd in commands:
        print(cmd)
        os.system(cmd)
    #output results dir to a file
    expe_setting = f"{args.dataset}"
    with open(f"{model_dir}/gge/commands_graphvae_gat.txt", 'a') as command_f:
        command_f.write(timestamp +': \n')
        command_f.write(expe_setting +'\n')
        for cmd in commands:
            command_f.write(cmd+'\n')
        command_f.write(f"############################################################################################## \n")    

    with open(f"{model_dir}/gge/results_graphvae_gat.txt", 'a') as result_f:
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