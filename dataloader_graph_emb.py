from node_classification.utils import process
import scipy.sparse as sp
import numpy as np
import torch
import math
import torch.nn.functional as F

import numpy as np
from ema_pytorch import EMA
from torch.utils.data import Dataset, DataLoader
from utils import trans_spcsr_to_torchcsr

def create_positional_embeddings(seq_len, emb_dim):
    """Create positional embeddings."""
    # Initialize the matrix with zeros
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))

    # Calculate positional encodings
    positional_embeddings = torch.zeros(seq_len, emb_dim)
    positional_embeddings[:, 0::2] = torch.sin(position * div_term)
    positional_embeddings[:, 1::2] = torch.cos(position * div_term)

    return positional_embeddings


#this method only for node with only one cluster
def get_neighbor_map_cluster(neighbor_map, cluster_index_list):
    #neighbor_map: original neighbor map
    #cluster_list: list of cluster index

    #neighbor_map = neighbor_map.cpu().numpy()

    indexes_incluster_list = []
    cluster_unique = np.unique(cluster_index_list, return_counts = True)
    for cluster_index in cluster_unique[0]:
        test = np.where(cluster_index_list == cluster_index)
        indexes_incluster_list.append(np.where(cluster_index_list == cluster_index)[0])

    max_num_node = cluster_unique[1].max()
    neighbor_map_cluster = np.zeros((len(cluster_index_list), max_num_node), dtype=int)
    
    for indexes_incluster in indexes_incluster_list:
        length_neighbor = len(indexes_incluster)
        neighbor_map_cluster[indexes_incluster, :length_neighbor] = neighbor_map[indexes_incluster,:][:, indexes_incluster].todense()  #get the adj in a cluster and pass it to the new adj neighbor_map_cluster
    
    return neighbor_map_cluster

#this method only for node with only one cluster
def get_neighbor_feats(neighbor_map, cluster_index_list, embeds_normalized):
    #get the index of nodes in each cluster
    indexes_incluster_list = []
    cluster_unique = np.unique(cluster_index_list, return_counts = True)
    for cluster_index in cluster_unique[0]:
        test = np.where(cluster_index_list == cluster_index)
        indexes_incluster_list.append(np.where(cluster_index_list == cluster_index)[0])

    #get the neighbor features in each cluster
    neighbor_feats = np.zeros_like(embeds_normalized, dtype=int)
    for indexes_incluster in indexes_incluster_list:
        seq_len = len(indexes_incluster)     #number of nodes in the cluster   
        emb_dim = embeds_normalized.shape[1]  # Embedding dimensions

        neighbor_map_incluster = neighbor_map[indexes_incluster,:][:, indexes_incluster]
        feats_incluster = embeds_normalized[indexes_incluster,:]
        norm_adj = process.normalize_adj(neighbor_map_incluster)
        norm_adj = torch.FloatTensor(norm_adj.todense())
        positional_embeddings_incluster = create_positional_embeddings(seq_len, emb_dim)

        neighbor_feats_incluster = torch.matmul(norm_adj, feats_incluster+positional_embeddings_incluster) 
        neighbor_feats[indexes_incluster, :] = neighbor_feats_incluster

    return neighbor_feats

def get_index_incluster_ingraph(cluster_indexes):
    #given same cluster_indexes,this method always produce the same results

    length = len(cluster_indexes)
    node_index_ingraph = np.arange(length)

    # Sort both arrays by the cluster_indexes to ensure elements of the same cluster are consecutive
    sort_idx = np.argsort(cluster_indexes)
    sorted_cluster_indexes = cluster_indexes[sort_idx]
    sorted_node_index = node_index_ingraph[sort_idx]
    
    # Find the uniquei ndexes and their starting indexes in the sorted array
    _, idx_start = np.unique(sorted_cluster_indexes, return_index=True)
    
    # Use the starting indexes to find the count of elements in each cluster
    counts = np.diff(np.append(idx_start, sorted_cluster_indexes.size))
    
    # Generate increasing indices for each cluster
    indices = np.concatenate([np.arange(count) for count in counts])
    
    # Initialize an array to hold the indices in original order
    original_order_indices = np.empty_like(indices)
    
    # Map the sorted indices back to the original order  :index of a node in its cluster
    original_order_indices[sort_idx] = indices
    node_index_incluster = original_order_indices

    #stack node_index_ingraph, cluster_indexes and original_order_indices
    merged_array = np.column_stack((node_index_ingraph, cluster_indexes, node_index_incluster))

    return merged_array

#dataset for training
class Large_Graph_Emb_Dataset(Dataset):  
    def __init__(self, feature_embs, cluster_indexes, labels, adj_matrix, neighbor_map_dim=1000):
        #feature_embs: node embeddings: torch.tensor
        #cluster_indexes: cluster index of each node: np.array
        #adj_matrix: adjacency matrix of the graph: sp.csr_matrix
        
        norm_adj = process.normalize_adj(adj_matrix)
        norm_adj = norm_adj.tocsr()
        norm_adj = trans_spcsr_to_torchcsr(norm_adj)
        self.feat_embs = torch.FloatTensor(feature_embs)

        #normalize feature embeddings
        self.data_min = self.feat_embs.min()
        self.data_max = self.feat_embs.max()
        self.feat_embs_norm = (self.feat_embs - self.data_min) / (self.data_max - self.data_min)
        data_len = self.feat_embs_norm.shape[0]  # Length of your data
        emb_dim = self.feat_embs_norm.shape[1]  # Embedding dimensions

        #create neighbor features for each node in overall graph
        positional_embeddings = create_positional_embeddings(data_len, emb_dim)
        self.all_neighbor_feats = torch.matmul(norm_adj, self.feat_embs_norm + positional_embeddings)

        self.index_dict = get_index_incluster_ingraph(cluster_indexes)         #

        cluster_num = len(np.unique(cluster_indexes))

        '''
        self.feat_embs_gt = torch.zeros((0, emb_dim))
        self.neighbor_feats_emb_gt = torch.zeros((0, emb_dim))
        self.neighbor_map_incluster_gt = torch.zeros((0, neighbor_map_dim))
        self.cluster_map_gt = torch.zeros((0, cluster_num))
        self.cluster_ind_of_neighbors = torch.zeros(0)
        self.labels = torch.zeros(0)
        
        
        for i in range(100):
            if i % 1000 == 0:
                import time
                time1 = time.time()
                print(f"{i}: {time1}")

            indices = np.where(adj_matrix[[i], :].todense()[0] == 1)[0]
            index_matrix = self.index_dict[indices]       #get the index information of neighbors in the cluster
            
            #get cluster map ground truth
            unique_cluster = np.unique(index_matrix[:, 1])    #get the cluster index of neighbors
            cluster_map = np.zeros((1, cluster_num))
            cluster_map[0, unique_cluster] = 1

            #get neighbor map ground truth in a cluster
            for cluster_index in unique_cluster:
                cluster_indices = np.where(index_matrix[:, 1] == cluster_index)[0]
                neighbor_indexes_incluster = index_matrix[cluster_indices, 2]
                neighbor_map_incluster = np.zeros((1, neighbor_map_dim))
                neighbor_map_incluster[0, neighbor_indexes_incluster] = 1

                self.cluster_map_gt = torch.cat((self.cluster_map_gt, torch.FloatTensor(cluster_map)), dim=0)       #cluster map for all neighbors of a node. if a node has nighbors in 3 clusters, 3 elements is 1
                self.neighbor_map_incluster_gt = torch.cat((self.neighbor_map_incluster_gt, torch.FloatTensor(neighbor_map_incluster)), dim=0)  #neighbor map for neighbors in this (cluster_index)th cluster. 
                self.feat_embs_gt = torch.cat((self.feat_embs_gt, self.feat_embs_norm[i:i+1]), dim=0)
                self.neighbor_feats_emb_gt = torch.cat((self.neighbor_feats_emb_gt, self.all_neighbor_feats[i:i+1]), dim=0)
                self.labels = torch.cat((self.labels, torch.IntTensor([labels[i]])), dim=0)

            self.cluster_ind_of_neighbors = torch.cat((self.cluster_ind_of_neighbors, torch.FloatTensor(unique_cluster)), dim=0)      #cluster indexes of neighbors of a node
        self.labels = self.labels.to(torch.int)
        self.cluster_ind_of_neighbors = self.cluster_ind_of_neighbors.to(torch.int)
        '''

        self.feat_embs_gt = []
        self.neighbor_feats_emb_gt = []
        self.neighbor_map_incluster_gt = []
        self.cluster_map_gt = []
        self.cluster_ind_of_neighbors = []
        self.labels = []
        
        for i in range(len(feature_embs)):
            indices = np.where(adj_matrix[[i], :].todense()[0] == 1)[1]
            index_matrix = self.index_dict[indices]       #get the index information of neighbors in the cluster
            
            #get cluster map ground truth
            unique_cluster = np.unique(index_matrix[:, 1])    #get the cluster index of neighbors
            cluster_map = np.zeros((1, cluster_num))
            cluster_map[0, unique_cluster] = 1

            #get neighbor map ground truth in a cluster
            for cluster_index in unique_cluster:
                cluster_indices = np.where(index_matrix[:, 1] == cluster_index)[0]
                neighbor_indexes_incluster = index_matrix[cluster_indices, 2]
                neighbor_map_incluster = np.zeros((1, neighbor_map_dim))
                neighbor_map_incluster[0, neighbor_indexes_incluster] = 1

                self.cluster_map_gt.append(torch.FloatTensor(cluster_map))       #cluster map for all neighbors of a node. if a node has nighbors in 3 clusters, 3 elements is 1
                self.neighbor_map_incluster_gt.append(torch.FloatTensor(neighbor_map_incluster))  #neighbor map for neighbors in this (cluster_index)th cluster. 
                self.feat_embs_gt.append(self.feat_embs_norm[i:i+1])
                self.neighbor_feats_emb_gt.append(self.all_neighbor_feats[i:i+1])
                self.labels.append(torch.IntTensor([labels[i]]))

            self.cluster_ind_of_neighbors.append(torch.FloatTensor(unique_cluster))     #cluster indexes of neighbors of a node
        
        self.cluster_map_gt= torch.stack(self.cluster_map_gt).squeeze()
        self.neighbor_map_incluster_gt = torch.stack(self.neighbor_map_incluster_gt).squeeze()
        self.feat_embs_gt = torch.stack(self.feat_embs_gt).squeeze()
        self.neighbor_feats_emb_gt = torch.stack(self.neighbor_feats_emb_gt).squeeze()
        self.labels = torch.stack(self.labels).squeeze()
        self.cluster_ind_of_neighbors = torch.cat(self.cluster_ind_of_neighbors, 0)

        self.labels = self.labels.to(torch.int)
        self.cluster_ind_of_neighbors = self.cluster_ind_of_neighbors.to(torch.int)
            
    def __len__(self):
        return len(self.feat_embs_gt)

    def __getitem__(self, idx):
        feature_emb = self.feat_embs_gt[idx]
        neighbor_feats_emb = self.neighbor_feats_emb_gt[idx]
        neighbor_map_incluster_gt = self.neighbor_map_incluster_gt[idx]
        cluster_map_gt = self.cluster_map_gt[idx]
        cluster_ind_of_neighbors = self.cluster_ind_of_neighbors[idx]
        label = self.labels[idx]

        return feature_emb, neighbor_feats_emb, neighbor_map_incluster_gt, cluster_map_gt, cluster_ind_of_neighbors, label

class Graph_Emb_Dataset(Dataset):  
    def __init__(self, feature_embs, labels, adj_matrix, pos_emb =1):
        #feature_embs: node embeddings: torch.tensor
        #adj_matrix: adjacency matrix of the graph: sp.csr_matrix

        #some adj_matrix, like citeseer, have values>1, we need to make it as 1
        adj_matrix[adj_matrix>1]=1
        
        norm_adj = process.normalize_adj(adj_matrix)
        norm_adj = torch.FloatTensor(norm_adj.todense())
        self.feat_embs = torch.FloatTensor(feature_embs)

        #normalize feature embeddings
        self.data_min = self.feat_embs.min()
        self.data_max = self.feat_embs.max()
        self.feat_embs_norm = (self.feat_embs - self.data_min) / (self.data_max - self.data_min)
        data_len = self.feat_embs_norm.shape[0]  # Length of your data
        emb_dim = self.feat_embs_norm.shape[1]  # Embedding dimensions

        #create neighbor features for each node in overall graph
        if pos_emb:
            positional_embeddings = create_positional_embeddings(data_len, emb_dim)
            self.all_neighbor_feats = torch.matmul(norm_adj, self.feat_embs_norm + positional_embeddings)
        else:
            self.all_neighbor_feats = torch.matmul(norm_adj, self.feat_embs_norm)

        self.labels = torch.FloatTensor(labels).to(torch.int)
        self.neighbor_map_gt = torch.FloatTensor(adj_matrix.todense())
        
        
    def __len__(self):
        return len(self.feat_embs_norm)

    def __getitem__(self, idx):
        return  self.feat_embs_norm[idx], self.all_neighbor_feats[idx], self.neighbor_map_gt[idx], self.labels[idx]
    
class Decoder_Dataset(Dataset):
    def __init__(self, data_dir):
        features = np.load(data_dir)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def load_graph_emb_dataset(params):
    feature_embs = np.load(params.datadir)
    labels = np.load(params.labeldir)

    dataset = params.dataset
    #adj, features, _, idx_train, idx_val, idx_test = process.load_data(dataset)
    graph, _, _, _, _, _, _= process.load(dataset)
    adj =convert_torchcoo_to_np_csr(graph.adjacency_matrix())
    adj_matrix = adj + sp.eye(adj.shape[0])

    graph_dataset = Graph_Emb_Dataset(feature_embs, labels, adj_matrix)
    dataloader = DataLoader(graph_dataset, batch_size=params.batchsize, shuffle=params.shuffle, num_workers=params.numworkers)
    return dataloader, graph_dataset

def load_large_graph_emb_dataset(params):
    feature_embs = np.load(params.datadir)
    cluster_indexes = np.load(params.clusterdir)     #cluster index of each node
    labels = np.load(params.labeldir)

    #TODO: here should read adj matrix data directly from a file
    adj = sp.load_npz(params.adjdir)
    #dataset = params.dataset
    #adj, features, _, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj_matrix = adj + sp.eye(adj.shape[0])

    large_graph_dataset = Large_Graph_Emb_Dataset(feature_embs, cluster_indexes, labels, adj_matrix, neighbor_map_dim=params.neighbor_map_dim)
    dataloader = DataLoader(large_graph_dataset, batch_size=params.batchsize, shuffle=params.shuffle, num_workers=params.numworkers)
    return dataloader, large_graph_dataset.index_dict

def load_large_graph_emb_dataset_for_encoding(params):
    #inorder to get the same number of latent z as original data.
    #load_large_graph_emb_dataset will get more than original.(because of cmap and nmap)
    feature_embs = np.load(params.datadir)
    cluster_indexes = np.load(params.clusterdir)     #cluster index of each node
    labels = np.load(params.labeldir)

    #TODO: here should read adj matrix data directly from a file
    adj = sp.load_npz(params.adjdir)
    #dataset = params.dataset
    #adj, features, _, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj_matrix = adj + sp.eye(adj.shape[0])

    graph_dataset = Large_Graph_Emb_Dataset_for_encoding(feature_embs, labels, adj_matrix)
    dataloader = DataLoader(graph_dataset, batch_size=params.batchsize, shuffle=params.shuffle, num_workers=params.numworkers)
    return dataloader, graph_dataset

def convert_torchcoo_to_np_csr(coo):
    torch_dense = coo.to_dense()
    np_sparse_csr = torch_dense.numpy()
    np_sparse_csr = sp.csr_matrix(np_sparse_csr)
    return np_sparse_csr

class Large_Graph_Emb_Dataset_for_encoding(Dataset):  
    def __init__(self, feature_embs, labels, adj_matrix, pos_emb =1):
        #feature_embs: node embeddings: torch.tensor
        #adj_matrix: adjacency matrix of the graph: sp.csr_matrix

        #some adj_matrix, like citeseer, have values>1, we need to make it as 1
        adj_matrix[adj_matrix>1]=1
        
        norm_adj = process.normalize_adj(adj_matrix)
        #norm_adj = torch.FloatTensor(norm_adj.todense())
        #because of OOM
        norm_adj = norm_adj.tocsr()
        norm_adj = trans_spcsr_to_torchcsr(norm_adj)
        self.feat_embs = torch.FloatTensor(feature_embs)

        #normalize feature embeddings
        self.data_min = self.feat_embs.min()
        self.data_max = self.feat_embs.max()
        self.feat_embs_norm = (self.feat_embs - self.data_min) / (self.data_max - self.data_min)
        data_len = self.feat_embs_norm.shape[0]  # Length of your data
        emb_dim = self.feat_embs_norm.shape[1]  # Embedding dimensions

        #create neighbor features for each node in overall graph
        if pos_emb:
            positional_embeddings = create_positional_embeddings(data_len, emb_dim)
            self.all_neighbor_feats = torch.matmul(norm_adj, self.feat_embs_norm + positional_embeddings)
        else:
            self.all_neighbor_feats = torch.matmul(norm_adj, self.feat_embs_norm)

        self.labels = torch.FloatTensor(labels).to(torch.int)
        #self.neighbor_map_gt = trans_spcsr_to_torchcsr(adj_matrix)
        
        
    def __len__(self):
        return len(self.feat_embs_norm)

    def __getitem__(self, idx):
        #return  self.feat_embs_norm[idx], self.all_neighbor_feats[idx], self.neighbor_map_gt[idx], self.labels[idx]
        return  self.feat_embs_norm[idx], self.all_neighbor_feats[idx], self.labels[idx]