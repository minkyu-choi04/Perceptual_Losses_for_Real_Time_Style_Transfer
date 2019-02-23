import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import os

def load_lsun(batch_size, img_size=256):
    #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/min/DATASETS/IMAGE/LSUN'), classes='train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                #normalize
                ]), target_transform=None),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/min/DATASETS/IMAGE/LSUN'), classes='val', transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                #normalize
                ]), target_transform=None),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/min/DATASETS/IMAGE/LSUN'), classes='test', transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                #normalize
                ]), target_transform=None),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, valid_loader, 10
