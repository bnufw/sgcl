import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    return classes, {name: index for index, name in enumerate(classes)}


def get_train_val_images_tiny(data_path):
    train_dir = os.path.join(data_path, "tiny-imagenet-200", "train")
    test_dir = os.path.join(data_path, "tiny-imagenet-200", "val")
    train_set = datasets.ImageFolder(train_dir)
    train_images = [path for path, _ in train_set.imgs]
    train_targets = np.array([label for _, label in train_set.imgs])
    _, class_to_idx = find_classes(train_dir)
    with open(os.path.join(test_dir, "val_annotations.txt")) as source:
        annotations = [line.split("\t") for line in source]
    class_map = {line[0]: line[1] for line in annotations}
    test_images, test_targets = [], []
    for name in sorted(os.listdir(os.path.join(test_dir, "images"))):
        if class_map[name] in class_to_idx:
            test_images.append(os.path.join(test_dir, "images", name))
            test_targets.append(class_to_idx[class_map[name]])
    return train_images, train_targets, test_images, np.array(test_targets), class_to_idx


def get_train_val_images_imagenet_subset(data_path):
    root = os.path.join(data_path, "imagenet_subset")
    train_dir = os.path.join(root, "train")
    train_set = datasets.ImageFolder(train_dir)
    train_images = [path for path, _ in train_set.imgs]
    train_targets = np.array([label for _, label in train_set.imgs])
    _, class_to_idx = find_classes(train_dir)
    test_images, test_targets = [], []
    with open(os.path.join(root, "val_100.txt")) as source:
        for line in source:
            path, label = line.rstrip().split(" ")
            test_images.append(os.path.join(root, path))
            test_targets.append(int(label))
    return train_images, train_targets, test_images, np.array(test_targets), class_to_idx


class ImageDataset(Dataset):
    def __init__(self, data, targets, class_to_idx, transform):
        self.data = data
        self.targets = targets
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], "rb") as source:
            image = Image.open(source).convert("RGB")
        return self.transform(image), self.targets[index]


def get_dataset(dataset_type, data_path):
    if dataset_type == "cifar100":
        train_transform = transforms.Compose([
            transforms.Pad(4), transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        return (datasets.CIFAR100(data_path, train=True, download=False, transform=train_transform),
                datasets.CIFAR100(data_path, train=False, download=False, transform=test_transform), 100)

    if dataset_type == "tiny-imagenet":
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_tiny(data_path)
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        classes = 200
    elif dataset_type == "imagenet-subset":
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_imagenet_subset(data_path)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        classes = 100
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_type))
    return (ImageDataset(train_data, train_targets, class_to_idx, train_transform),
            ImageDataset(test_data, test_targets, class_to_idx, test_transform), classes)
