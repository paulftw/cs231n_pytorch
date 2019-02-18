import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import random

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img
    plt.imshow(img)
    plt.show()

# Manhattan distance
def manhattan(a, b):
    return abs(a.astype(int) - b.astype(int)).sum()


# Euclidian distance
def euclidian(a, b):
    dif = a.astype(int) - b.astype(int)
    return np.multiply(dif, dif).sum()

def nn_classify(img, distance_fn):
    best_distance = 1e100
    best_idx = 0
    for idx, t in enumerate(trainset.train_data):
        d = distance_fn(t, img)
        if d < best_distance:
            best_distance = d
            best_idx = idx

    print(best_idx)
    return trainset.train_labels[best_idx]

itoc = random.choice(testset.test_data)
res = nn_classify(itoc, manhattan)
print(classes[res])
imshow(itoc)
