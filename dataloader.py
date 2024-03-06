import os 
import torch, torchvision
from torchvision import transforms


def check_for_download(return_dataloader):
    def download_if_empty(path='./data/mnist'):
        path_exists = return_dataloader()
        if path_exists == "True":
            transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])
            mnist_data_train = torchvision.datasets.MNIST(path, download=False, train=True, transform=transform)
            mnist_data_test = torchvision.datasets.MNIST(path, download=False, train=False, transform=transform)
            return torch.utils.data.DataLoader(mnist_data_train,
                                               batch_size = 4,
                                               shuffle=True,
                                               num_workers=4,),torch.utils.data.DataLoader(mnist_data_test,
                                               batch_size = 4,
                                               shuffle=True,
                                               num_workers=4,)
    
        else:
            transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])
            mnist_data_train = torchvision.datasets.MNIST(path, download=True, train=True, transform=transform)
            mnist_data_test = torchvision.datasets.MNIST(path, download=True, train=False, transform=transform)
            return torch.utils.data.DataLoader(mnist_data_train,
                                               batch_size = 4,
                                               shuffle=True,
                                               num_workers=4),torch.utils.data.DataLoader(mnist_data_test,
                                               batch_size = 4,
                                               shuffle=True,
                                               num_workers=4)
    return download_if_empty
 

@check_for_download
def return_dataloader(path='./data'):
    if os.path.exists(path):
        return True
    return False
