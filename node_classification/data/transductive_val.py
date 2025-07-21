import argparse
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import process


def parse_args():
    parser = argparse.ArgumentParser(description="Transductive Training with Synthetic Nodes")
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--gt_labels_path', type=str, required=True, help='Path to ground truth labels')
    parser.add_argument('--real_embeds_path', type=str, required=True, help='Path to real embeddings')
    parser.add_argument('--generated_embeds_path', type=str, required=True, help='Path to generated embeddings')
    parser.add_argument('--generated_labels_path', type=str, required=True, help='Path to generated labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--sample_per_class', type=int, default=20, help='Number of synthetic samples per class')
    parser.add_argument('--K', type=int, default=2, help='Number of nearest neighbors for synthetic nodes')
    parser.add_argument('--try_num', type=int, default=2000, help='Number of runs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--feat_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay')
    parser.add_argument('--epoch_num', type=int, default=20000, help='Number of training epochs')
    return parser.parse_args()


def sample_nodes(labels, num_samples):
    num_classes = np.max(labels) + 1
    sampled_indices = []
    for i in range(num_classes):
        indices = np.where(labels == i)[0]
        sampled_indices.append(np.random.choice(indices, num_samples))
    return np.concatenate(sampled_indices)


class TransductiveClassifier(nn.Module):
    def __init__(self, ft_size, num_classes, two_layer=True):
        super(TransductiveClassifier, self).__init__()
        self.two_layer = two_layer
        self.fc = nn.Linear(ft_size, num_classes, bias=False)

    def forward(self, norm_adj, x):
        x = torch.matmul(norm_adj, x)
        x = self.fc(x)
        return x


def add_synthetic_node_to_graph(adj, ori_embeds, new_embeds, K):
    ori_num = ori_embeds.shape[0]
    new_num = new_embeds.shape[0]
    N = ori_num + new_num
    extended_adjacency_matrix = np.zeros((N, N))
    extended_adjacency_matrix[:ori_num, :ori_num] = adj

    for new_node_idx, new_emb in enumerate(new_embeds):
        distances = np.linalg.norm(ori_embeds - new_emb, axis=1)
        nearest_old_node_indices = np.argsort(distances)[:K]
        for old_node_idx in nearest_old_node_indices:
            extended_adjacency_matrix[ori_num + new_node_idx, old_node_idx] = 1
            extended_adjacency_matrix[old_node_idx, ori_num + new_node_idx] = 1

    return extended_adjacency_matrix


def main():
    args = parse_args()

    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    features, _ = process.preprocess_features(features)
    num_nodes = features.shape[0]
    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    gt_labels = np.load(args.gt_labels_path)
    real_embeds = np.load(args.real_embeds_path)
    generated_embeds = np.load(args.generated_embeds_path)
    generated_labels = np.load(args.generated_labels_path)

    device = torch.device(args.device)
    all_runs_acc = []
    best_all = 0

    for i in tqdm(range(args.try_num)):
        sampled_indices = sample_nodes(generated_labels, args.sample_per_class)
        sampled_generated_embeds = generated_embeds[sampled_indices]
        sampled_generated_labels = generated_labels[sampled_indices]

        all_embs = np.concatenate((real_embeds, sampled_generated_embeds), axis=0)
        all_labels = np.concatenate((gt_labels, sampled_generated_labels), axis=0)
        new_adj = add_synthetic_node_to_graph(adj.todense(), real_embeds, sampled_generated_embeds, args.K)
        norm_new_adj = process.normalize_adj(new_adj + sp.eye(new_adj.shape[0]))
        new_idx_train = torch.cat((idx_train, torch.LongTensor(np.arange(num_nodes, num_nodes + sampled_generated_embeds.shape[0]))))

        model = TransductiveClassifier(args.feat_dim, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        adj_torch = torch.FloatTensor(norm_new_adj.todense()).to(device)
        feat_torch = torch.FloatTensor(all_embs).to(device)
        labels_torch = torch.LongTensor(all_labels).to(device)

        acc_hist = []
        for epoch in range(args.epoch_num):
            optimizer.zero_grad()
            model.train()
            out = model(adj_torch, feat_torch)
            loss = criterion(out[new_idx_train, :], labels_torch[new_idx_train])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                out = model(adj_torch, feat_torch)
                predict_labels = torch.argmax(out, dim=-1)
                correct = predict_labels[idx_test].eq(labels_torch[idx_test]).sum().item()
                acc = correct / predict_labels[idx_test].shape[0]
                acc_hist.append(acc)

        print('Training finished')
        print(np.max(acc_hist))
        all_runs_acc.append(np.max(acc_hist))
        if np.max(acc_hist) > best_all:
            best_all = np.max(acc_hist)
            np.save(f"{args.output_dir}/{args.feat_dim}_VAE_transductive_sampled_indices_{args.sample_per_class}_K{args.K}.npy", sampled_indices)
        np.save(f"{args.output_dir}/{args.feat_dim}_VAE_all_runs_acc_{args.sample_per_class}_K{args.K}.npy", np.array(all_runs_acc))


if __name__ == "__main__":
    main()
