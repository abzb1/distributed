import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torchvision as torchvision
from torchvision import datasets
import torchvision.transforms as transforms

def cifar10data(batch_size, num_workers):
    path = "/home/ohs/dataset/cifar10/"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
         ])

    training_data = datasets.CIFAR10(
        root = path,
        train = True,
        download = False,
        transform = transform)

    test_data = datasets.CIFAR10(
        root = path,
        train = False,
        download = False,
        transform = transform)

    train_dataloader = DataLoader(training_data, batch_size = batch_size, num_workers = num_workers, drop_last = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, drop_last = True)

    return train_dataloader, test_dataloader

def imageNetdata(batch_size, num_workers, epochs, **kwargs):
    path = "/home/ohs/dataset/imagenet/"

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop((224,224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transform = transforms.Compose(
        [transforms.Resize((256,256)),
         transforms.CenterCrop((224,224)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    training_data = datasets.ImageNet(
        root = path,
        split = "train",
        transform = train_transform)

    test_data = datasets.ImageNet(
        root = path,
        split = "val",
        transform = test_transform)

    train_sampler = DistributedSampler(training_data)
    train_sampler.set_epoch(epoch=epochs)
    test_sampler = DistributedSampler(test_data)

    train_dataloader = DataLoader(training_data, batch_size = batch_size, num_workers = num_workers, drop_last = True, sampler=train_sampler, shuffle=train_sampler is None)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, drop_last = True, sampler=test_sampler, shuffle=False)

    return train_dataloader, test_dataloader

