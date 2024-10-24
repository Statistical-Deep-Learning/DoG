import math
import numpy as np
import socket
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        
def betas_for_alpha_bar(num_diffusion_timesteps:int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)




def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.



def plot_tsne(gt_embs, gt_labels, gen_embs, gen_labels, save_dir, epoch, desc=None):
    #all inputs are numpy
    combined_embeds = np.concatenate((gt_embs, gen_embs), axis=0)
    combined_labels = np.concatenate((gt_labels, gen_labels), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
    tsne_results = tsne.fit_transform(combined_embeds)
    plt.figure(figsize=(20, 12))

    n_real = gt_embs.shape[0]

    tsne_results_real = tsne_results[:n_real]
    tsne_results_generated = tsne_results[n_real:]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
'#ac2f75','#c043fb','#c36709','#d315f2','#245746','#d8588c','#3ac1e6','#2757ae','#5851a5','#194d48','#099473','#d0f3c5','#fe4faf','#c05263','#d8b1f3','#1d9393',
'#8ea720','#c109b9','#7f201f','#caf497',]

    unique_labels = np.unique(np.concatenate((gt_labels, gen_labels)))  # Get all unique labels from both sets

    for i in unique_labels:  # Iterate through each unique label
    # Plot real data
        plt.scatter(tsne_results_real[gt_labels == i, 0], tsne_results_real[gt_labels == i, 1], 
                color=colors[i % len(colors)], label=f'Real - Label {i}', s=5)
    
    # Plot generated data
        plt.scatter(tsne_results_generated[gen_labels == i, 0], tsne_results_generated[gen_labels == i, 1], 
                color=colors[i % len(colors)], label=f'Generated - Label {i}', marker='>', s=40)


        plt.legend(fontsize='small')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.title('t-SNE visualization of embeddings')
        plt.savefig(os.path.join(save_dir,f'tsne_{epoch}_{desc}.png'))

def trans_spcsr_to_torchcsr(sparse_matrix):
    values = torch.tensor(sparse_matrix.data)
    indices = torch.tensor(sparse_matrix.indices)
    indptr = torch.tensor(sparse_matrix.indptr)
    shape = sparse_matrix.shape

    # Create the sparse tensor in PyTorch
    sparse_tensor = torch.sparse_csr_tensor(indptr, indices, values, size=shape, dtype=torch.float32)
    return sparse_tensor

if __name__ == '__main__':
    #gt_labels = np.load('/home/local/ASUAD/ywan1053/gcn/MERIT-main/embed/cora/all_gt_labels.npy')
    gt_labels = np.load('/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy')

    print(gt_labels.shape)
    all_embeds = np.load('/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy')
    print(all_embeds.shape)
    generated_lables = np.load('/data-drive/backup/changyu/expe/gge/unet_1d_core512_all_norm_ema/labels_2730_diffusion_3000_1.8.npy')
    print(generated_lables.shape)
    generated_embeds = np.load('/data-drive/backup/changyu/expe/gge/unet_1d_core512_all_norm_ema/samples_2730_diffusion_3000_1.8.npy')
    print(generated_embeds.shape)
    plot_tsne(all_embeds, gt_labels, generated_embeds, generated_lables, '/home/local/ASUAD/changyu2/generate_graph_embbedding', 222222)
    print("finished")


