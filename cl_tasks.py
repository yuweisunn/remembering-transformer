import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader, Subset

#def convert_to_three_channel(image):
#    return torch.cat([image, image, image], dim=0)

def split_dataset(trainset, n_splits, type, dataset, class_li):
    """ Split the training set into multiple subsets based on class labels. """
    try:
        train_x = trainset.data
    except:
        train_x = np.array(trainset.imgs)

    train_y = np.array(trainset.targets)

    # Generate random permutation indices
    permutation_indices = np.random.permutation(len(train_x))

    # Shuffle train_x and train_y arrays
    train_x = train_x[permutation_indices]
    train_y = train_y[permutation_indices]

    if dataset == 'tin':
        n_classes = 200 
        random_indices = np.random.choice(len(train_x), size=1000, replace=False)
        train_x = train_x[random_indices]
        train_y = train_y[random_indices]
        
    else:
        n_classes = len(trainset.classes)

    if n_classes % n_splits != 0:
        print("n_classes should be a multiple of the number of splits!")
        raise NotImplementedError
    
    class_for_split = n_classes // n_splits
    trainset_splits = []
    
    for i in range(n_splits):
        start = i * class_for_split
        end = (i + 1) * class_for_split
        split_idxs = np.where(np.in1d(train_y, class_li[start:end]))[0] #np.logical_and(
        split_data = train_x[split_idxs]
        split_labels = train_y[split_idxs]
        
        
        if dataset == 'cifar10': 
            if type == 'train':
                train_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                split_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
            else:
                test_transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                split_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        
        elif dataset == 'mnist': 
            if type == 'train':
               
                transform_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
                split_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=False, transform=transform_train)
            else:
                transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
                split_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=False, transform=transform_test)


        elif dataset == 'cifar100': 
            if type == 'train':
                transform_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                split_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=False, transform=transform_train)
            else:
                transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                split_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=False, transform=transform_test)


        elif dataset == 'tin':
            if type == 'train':
                transform_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                split_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train',transform=transform_train)
            
            else:   
                transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                split_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/test',transform=transform_test)
 
        try:
            split_dataset.data = split_data
        except:
            split_dataset.imgs = split_data

        split_dataset.targets = split_labels
       
        trainset_splits.append(split_dataset)

    return trainset_splits

 
def permute(datasets):
    """ Given the training set, permute pixels of each image the same way. """
    h = w = 28 #32
    #c = 3  # Number of channels
    perm_inds = list(range(h * w))
    np.random.shuffle(perm_inds)

    perm_datasets = []
    for dataset in datasets:
        num_images = dataset.shape[0]
        flat_dataset = dataset.reshape(num_images, h * w)
        permuted_dataset = flat_dataset[:, perm_inds].reshape(num_images, h, w)
        perm_datasets.append(permuted_dataset)

    return perm_datasets
