from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import ml_collections

data_config_default = ml_collections.ConfigDict()
data_config_default.data_dir = '/data-drive/backup/changyu'
data_config_default.train_batchsize = 64
data_config_default.test_batchsize = 64
data_config_default.numworkers = 8



def load_data(params):
    data_config = data_config_default
    data_config.train_batchsize = params.batchsize
    
    mnist_transform = transforms.Compose([
      transforms.ToTensor(),
      #transforms.Normalize((0.5,), (0.5,))  # Normalizing to [-1, 1]
    ])
    train_dataset = MNIST(root = data_config.data_dir, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(root = data_config.data_dir, transform=mnist_transform, train=False, download=True)
    trainloader = DataLoader(
                    train_dataset,
                    batch_size = data_config.train_batchsize,
                    num_workers = data_config.numworkers,
                    drop_last = True
                )
    testloader = DataLoader(
                    test_dataset,
                    batch_size = data_config.test_batchsize,
                    num_workers = data_config.numworkers,
                    drop_last = True
                )
    return trainloader, testloader

def transback(data):
    return data / 2 + 0.5