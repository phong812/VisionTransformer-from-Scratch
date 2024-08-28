import torch
import torchvision

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os


def get_data_loaders(train_dir, test_dir, transform: transforms.Compose, batch_size: int, num_workers: int):
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
    )

    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    
    class_names = train_data.classes
    
    return train_dataloader, test_dataloader, class_names

if __name__ == "__main__":
    #data_path = "../data/food-101-tiny"
    data_path = "C:/Users/Gia Phong/Downloads/ViT/data/food-101-tiny"
    train_path = data_path + "/train"
    test_path = data_path + "/valid"
    train_dataloader, _, class_names = get_data_loaders(train_path, test_path, transforms.ToTensor(), 1, 4)
    print(len(train_dataloader))
    print(len(class_names))
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.size(), train_labels.size())


