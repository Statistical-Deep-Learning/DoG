import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ml_collections

data_config_default = ml_collections.ConfigDict()
data_config_default.data_dir = '/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy'
data_config_default.data_labels_dir = '/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_gt_labels.npy'
data_config_default.train_batchsize = 128
data_config_default.numworkers = 8
data_config_default.norm = 0



class Emb_Dataset(Dataset):
    def __init__(self, emb_dir, label_dir, norm):
        self.norm = norm
        self.emb = np.load(emb_dir)        
        self.max = self.emb.max()
        self.min = self.emb.min()
        if self.norm:
            self.emb = self.normalize(self.emb)
        self.labels = np.load(label_dir).reshape(-1,1)
        self.emb = torch.from_numpy(self.emb).float()
        self.labels = torch.from_numpy(self.labels)
        B, D = self.emb.shape
        self.emb = self.emb.unsqueeze(1)
        self.labels = self.labels.squeeze(1)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        emb = self.emb[idx]
        label = self.labels[idx]

        return emb, label
    
    def normalize(self, data):
        return (data-self.min)/(self.max - self.min)
    
    def transback(self, data):
        return data*(self.max - self.min) + self.min

def load_data(params):
    data_config = data_config_default
    data_config.train_batchsize = params.batchsize
    data_config.data_dir = params.datadir
    data_config.data_labels_dir = params.labeldir
    data_config.norm = params.norm

    train_dataset = Emb_Dataset(emb_dir=data_config.data_dir, label_dir=data_config.data_labels_dir, norm = data_config.norm)
    trainloader = DataLoader(
                    train_dataset,
                    batch_size = data_config.train_batchsize,
                    num_workers = data_config.numworkers,
                )

    return trainloader, train_dataset

class Decode_Dataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

if __name__ == '__main__':
    ds = Emb_Dataset(emb_dir=data_config_default.data_dir, label_dir=data_config_default.data_labels_dir, norm=1)
    ds_unnorm = ds.transback(ds.emb)

    print(ds_unnorm)