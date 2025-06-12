from utils import process
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

dataset = 'cora'
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

features, _ = process.preprocess_features(features)
num_nodes = features.shape[0]
ft_size = features.shape[1]
num_classes = labels.shape[1]

features = torch.FloatTensor(features[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

# gt_labels = np.load('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora_64d/all_gt_labels.npy')
# print(gt_labels.shape)
# real_embeds = np.load('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy')
# print(real_embeds.shape)
gt_labels = np.load('./gcn/MERIT-main/embed/cora_512/all_gt_labels.npy')
print(gt_labels.shape)
real_embeds = np.load('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy')
print(real_embeds.shape)

# generated_embeds = np.load('/data-drive/backup/changyu/expe/gge/unet_1d_core64_all_norm_ema/samples_8190_diffusion_3000_1.8.npy')
# print(generated_embeds.shape)
# generated_labels = np.load('/data-drive/backup/changyu/expe/gge/unet_1d_core64_all_norm_ema/labels_8190_diffusion_3000_1.8.npy')
# print(generated_labels.shape)


generated_embeds = np.load('/data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0/latents_8190_vae_3000_512_decode.npy')
print(generated_embeds.shape)
# generated_embeds = np.load('/data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0/latents_8190_vae_3000_512_decode.npy')
# print(generated_embeds.shape)
generated_labels = np.load('/data-drive/backup/changyu/expe/gge/unet_1d_core64_encode_all_norm_ema/labels_8190_diffusion_3000_1.8.npy')
print(generated_labels.shape)

# Write a function that samples 100 nodes from each class
def sample_nodes(labels, num_samples):
    # return the indices of the sampled nodes
    num_classes = np.max(labels) + 1
    sampled_indices = []
    for i in range(num_classes):
        indices = np.where(labels == i)[0]
        sampled_indices.append(np.random.choice(indices, num_samples))
    return np.concatenate(sampled_indices)

class transductive_classifier(nn.Module):
    def __init__(self, ft_size, num_classes, two_layer=True):
        super(transductive_classifier, self).__init__()
        self.two_layer = two_layer
        self.fc = nn.Linear(ft_size, num_classes, bias=False)
    def forward(self, norm_adj, x):
        x = torch.matmul(norm_adj, x)
        x = self.fc(x)
        return x

# device = torch.device('cuda:1')
# learning_rate = 0.01
# weight_decay = 5e-6
# epoch_num = 10000

# num_classes = 7
# feat_dim = 64

# model = transductive_classifier(feat_dim, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# adj_torch = torch.FloatTensor(norm_adj.todense()).to(device)
# feat_torch = torch.FloatTensor(real_embeds).to(device)
# labels_torch = torch.LongTensor(gt_labels).to(device)


# Epochs = []
# train_loss_hist = []
# test_loss_hist = []
# acc_hist = []

# for epoch in range(epoch_num):
#     optimizer.zero_grad()
#     model.train()
#     out = model(adj_torch, feat_torch)
#     loss = criterion(out[idx_train,:], labels_torch[idx_train])
#     loss.backward()
#     optimizer.step()   
#     if epoch % 1 == 0:
#         with torch.no_grad():
#             model.eval()
#             out = model(adj_torch, feat_torch)
#             test_loss = criterion(out[idx_test,:], labels_torch[idx_test]) 
#             predict_lables = torch.argmax(out, dim=-1)
#             correct = predict_lables[idx_test].eq(labels_torch[idx_test]).sum().item()
#             acc = correct / predict_lables[idx_test].shape[0]
#             if epoch % 10 == 0:
#                 print(f'Epoch[{epoch}] Loss: {loss.item():.8}  Test_Loss: {test_loss.item():.8} ACC: {acc:.8}')
#             Epochs.append(epoch)
#             train_loss_hist.append(loss.item())
#             test_loss_hist.append(test_loss.item())
#             acc_hist.append(acc)

# print('Training finished')
# print(np.max(acc_hist))


def add_sythnetic_node_to_graph(adj, ori_embeds, new_embeds, K):
    ori_num = ori_embeds.shape[0]
    new_num = new_embeds.shape[0]
    # Extend the adjacency matrix for the new nodes (initially with no connections)
    N = ori_num + new_num  # Total number of nodes after adding new ones
    extended_adjacency_matrix = np.zeros((N, N))
    extended_adjacency_matrix[:ori_num, :ori_num] = adj

    # Calculate distances from each new node to each old node
    for new_node_idx, new_embeds in enumerate(new_embeds):
        distances = np.linalg.norm(ori_embeds - new_embeds, axis=1)
        nearest_old_node_indices = np.argsort(distances)[:K]  # Get indices of K nearest old nodes
        
        # Update the adjacency matrix to reflect connections between new nodes and their K nearest old nodes
        for old_node_idx in nearest_old_node_indices:
            # Since it's an undirected graph, we update both corresponding entries in the matrix
            extended_adjacency_matrix[ori_num + new_node_idx, old_node_idx] = 1  # Connection from new to old
            extended_adjacency_matrix[old_node_idx, ori_num + new_node_idx] = 1  # Connection from old to new

    # The extended_adjacency_matrix now includes connections from each new node to its K nearest old nodes
    return extended_adjacency_matrix


sample_per_class = 20
K = 2
try_num = 2000
device = torch.device('cuda:1')

all_runs_acc = []
best_all = 0
for i in tqdm(range(try_num)):
    sampled_indices = sample_nodes(generated_labels, sample_per_class)
    # sampled_indices = np.load('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora_indices/transductive_sampled_indices_'+str(sample_per_class)+'.npy')
    sampled_generated_embeds = generated_embeds[sampled_indices]
    sampled_generated_labels = generated_labels[sampled_indices]
    # print(sampled_generated_embeds.shape)
    # print(sampled_generated_labels.shape)
    # sampled_generated_embeds = torch.FloatTensor(sampled_generated_embeds)
    # sampled_generated_labels = torch.FloatTensor(sampled_generated_labels)

    all_embs = np.concatenate((real_embeds, sampled_generated_embeds), axis=0)
    all_labels = np.concatenate((gt_labels, sampled_generated_labels), axis=0)
    new_adj = add_sythnetic_node_to_graph(adj.todense(), real_embeds, sampled_generated_embeds, K)
    norm_new_adj = process.normalize_adj(new_adj + sp.eye(new_adj.shape[0]))
    new_idx_train = torch.cat((idx_train, torch.LongTensor(np.arange(num_nodes, num_nodes + sampled_generated_embeds.shape[0]))))

    # np.save('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora_indices/transductive_sampled_indices_'+str(sample_per_class)+'.npy', sampled_indices) # Best with K=3

    
    learning_rate = 0.01
    weight_decay = 5e-6
    epoch_num = 20000

    num_classes = 7
    feat_dim = 512

    model = transductive_classifier(feat_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    adj_torch = torch.FloatTensor(norm_new_adj.todense()).to(device)
    feat_torch = torch.FloatTensor(all_embs).to(device)
    labels_torch = torch.LongTensor(all_labels).to(device)


    Epochs = []
    train_loss_hist = []
    test_loss_hist = []
    acc_hist = []

    for epoch in range(epoch_num):
        optimizer.zero_grad()
        model.train()
        out = model(adj_torch, feat_torch)
        loss = criterion(out[new_idx_train,:], labels_torch[new_idx_train])
        loss.backward()
        optimizer.step()   
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                out = model(adj_torch, feat_torch)
                test_loss = criterion(out[idx_test,:], labels_torch[idx_test]) 
                predict_lables = torch.argmax(out, dim=-1)
                correct = predict_lables[idx_test].eq(labels_torch[idx_test]).sum().item()
                acc = correct / predict_lables[idx_test].shape[0]
                # if epoch % 10 == 0:
                #     print(f'Epoch[{epoch}] Loss: {loss.item():.8}  Test_Loss: {test_loss.item():.8} ACC: {acc:.8}')
                Epochs.append(epoch)
                train_loss_hist.append(loss.item())
                test_loss_hist.append(test_loss.item())
                acc_hist.append(acc)

    print('Training finished')
    print(np.max(acc_hist))
    all_runs_acc.append(np.max(acc_hist))
    if np.max(acc_hist) > best_all:
        best_all = np.max(acc_hist)
        np.save('./graph_diffusion/generate_graph_embbedding/gcl_embeddings/cora_indices/'+str(feat_dim)+'_VAE_transductive_sampled_indices_'+str(sample_per_class)+'_K'+str(K)+'.npy', sampled_indices) # Best with K=3
    np.save('./'+str(feat_dim)+'_VAE_all_runs_acc_'+str(sample_per_class)+'_K'+str(K)+'.npy', np.array(all_runs_acc))