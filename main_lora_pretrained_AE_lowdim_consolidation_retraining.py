# Author: Yuwei Sun
# 2024/04/15

 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from moe_lora_customer_autoencoder import MoE
import torch.optim as optim
import time, os
from timm.scheduler import CosineLRScheduler
import numpy as np
import loralib as lora
from transformers import ViTForImageClassification
from tqdm import tqdm
from cl_tasks import *
import argparse
from torchvision import transforms, datasets
import copy
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, BertModel


def print_trainable_status(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

class MMD_loss(nn.Module):
  def __init__(self, kernel_mul = 2.0, kernel_num = 5):
      super(MMD_loss, self).__init__()
      self.kernel_num = kernel_num
      self.kernel_mul = kernel_mul
      self.fix_sigma = None
      return

  def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
      n_samples = int(source.size()[0])+int(target.size()[0])
      total = torch.cat([source, target], dim=0)

      total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
      total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
      L2_distance = ((total0-total1)**2).sum(2) 
      if fix_sigma:
        bandwidth = fix_sigma
      else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
      bandwidth /= kernel_mul ** (kernel_num // 2)
      bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
      kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
      return sum(kernel_val)

  def forward(self, source, target):
      batch_size = int(source.size()[0])
      kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
      XX = kernels[:batch_size, :batch_size]
      YY = kernels[batch_size:, batch_size:]
      XY = kernels[:batch_size, batch_size:]
      YX = kernels[batch_size:, :batch_size]
      loss = torch.mean(XX + YY - XY -YX)
      return loss
  

class Autoencoder(nn.Module):

    def __init__(self, input_dims, code_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, input_dims),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


class ModularNN(nn.Module):
    def __init__(self,k,num_expert,ff):
        super(ModularNN, self).__init__()
       
        self.moe = MoE(
                    input_size = dim*hidden_dim, 
                    hidden_dim = hidden_dim,
                    dim = dim,
                    num_experts = num_expert,       # increase the experts (# parameters) of your model without increasing computation
                    ff=ff)

    def forward(self, input_data, head_mask = None, output_attentions = False):
        input_data = input_data.reshape(-1,dim*hidden_dim)
 
        # training
        if torch.is_grad_enabled():
            out_moe, aux_loss = self.moe(input_data, gate)

        # evaluation
        else:
            out_moe, aux_loss = self.moe(input_data, gate_eval)

        out_moe = out_moe.reshape(-1,dim,hidden_dim)
        
        return (out_moe, aux_loss)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='cifar10 or cifar100 or tin or imagenet (default: cifar10)')
parser.add_argument('--mode', type=str, default="train",
                    help='train or eval (default: train)')
parser.add_argument('--model_size', type=str, default="base",
                    help='base or large (default: base)')
parser.add_argument('--task_type', type=str, default="split",
                    help='normal or split or permute or datasets (default: split)')
parser.add_argument('--epochs', type=int, default=100,
                    help='epochs (default: 100)')
parser.add_argument('--n_k', type=int, default=1,
                    help='n_k (default: 1)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=None,
                    help='test batch size (default: None)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--retrain', action='store_true',
                        help='retrain (default: False)')
parser.add_argument('--limem', type=int, default=0,
                    help='memory slots (default: 0)') 
parser.add_argument('--num_task', type=int, default=5,
                    help='num of tasks (default: 5)') 
parser.add_argument('--vae_epochs', type=int, default=10,
                    help='vae epochs (default: 10)')
parser.add_argument('--vae_lr', type=float, default=5e-3,
                    help='AE learning rate (default: 5e-3)')
parser.add_argument('--seed', type=int, default=0,
                    help='seed (default: 0)')

args = parser.parse_args()

dataset = args.dataset
mode = args.mode 
model_size = args.model_size 
task_type = args.task_type
epochs = args.epochs 
n_k = args.n_k
batch_size = args.batch_size
lr = args.lr
retrain = args.retrain
seed = args.seed
num_task = args.num_task
vae_epochs = args.vae_epochs
vae_lr = args.vae_lr
limem = args.limem
num_expert = num_task
test_batch_size = args.test_batch_size

# when the memory is not limited 
if not limem:
    limem = num_task

if not test_batch_size:
    test_batch_size = batch_size


# set the modality of the dataset
if dataset in ['mnist', 'cifar10', 'cifar100', 'tin']:
    modality = 'vision'
else:
    modality = 'nlp'

# Set the random seed
torch.manual_seed(seed)
np.random.seed(seed)

epsilon = -1
herd_size = 512
alpha = -1


if task_type == "datasets":

        image_size = 224
        patch_size = 16  
        dim = int(image_size/patch_size)**2 + 1
        trainloader_splits = []
        testloader_splits = []
        
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Use only the first n samples (memory constraint)
        subset_indices = range(5000)

        trainset_food = torchvision.datasets.Food101(root='./data', split='train',
                                                download=True, transform=transform_train)
        #trainset_food = Subset(trainset_food, subset_indices)
        testset_food = torchvision.datasets.Food101(root='./data', split='test',
                                                download=True, transform=transform_test)
        #testset_food = Subset(testset_food, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_food, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_food, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))
        
        trainset_flower = torchvision.datasets.Flowers102(root='./data', split='train',
                                                download=False, transform=transform_train)
        #trainset_flower = Subset(trainset_flower, subset_indices)
        testset_flower = torchvision.datasets.Flowers102(root='./data', split='test',
                                                download=False, transform=transform_test)
        #testset_flower = Subset(testset_flower, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_flower, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_flower, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))

        trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform_train)
        trainset_cifar10 = Subset(trainset_cifar10, subset_indices)
        testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=False, transform=transform_test)
        #testset_cifar10 = Subset(testset_cifar10, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_cifar10, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_cifar10, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))
        
        trainset_air = torchvision.datasets.FGVCAircraft(root='./data',split='train',
                                                download=False, transform=transform_train)
        #trainset_air = Subset(trainset_air, subset_indices)
        testset_air = torchvision.datasets.FGVCAircraft(root='./data', split='val',
                                                download=False, transform=transform_test)
        #testset_air = Subset(testset_air, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_air, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_air, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))
        
        trainset_pet = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval',
                                                download=False, transform=transform_train)
        #trainset_pet = Subset(trainset_pet, subset_indices)
        testset_pet = torchvision.datasets.OxfordIIITPet(root='./data', split='test',
                                                download=False, transform=transform_test)
        #testset_pet = Subset(testset_pet, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_pet, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_pet, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))
        
        trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=False, transform=transform_train)
        trainset_cifar100 = Subset(trainset_cifar100, subset_indices)
        testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=False, transform=transform_test)
        #testset_cifar100 = Subset(testset_cifar100, subset_indices)
        trainloader_splits.append(torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True))
        testloader_splits.append(torch.utils.data.DataLoader(testset_cifar100, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True))
         
else:
    if dataset == "mnist":
        image_size = 224
        num_classes = 10
        patch_size = 16  
        dim = int(image_size/patch_size)**2 + 1
        code_dim = 1
        
        transform_train = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
         
        transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=False, transform=transform_train)
        
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=False, transform=transform_test)
        
        if task_type == 'normal':
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)


        if task_type == 'split':
            class_li = list(range(num_classes))
            np.random.shuffle(class_li)

            trainset_splits = split_dataset(trainset, n_splits=num_task, type = 'train', dataset = dataset, class_li = class_li)
            trainloader_splits = []
            for i, task in enumerate(trainset_splits):
                trainloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
            
            testset_splits = split_dataset(testset, n_splits=num_task, type = 'test', dataset = dataset, class_li = class_li)
            testloader_splits = []
            for i, task in enumerate(testset_splits):
                testloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


        if task_type == 'permute':
            trainloader_splits = []
            testloader_splits = []
            train_len = len(trainset)//num_task
            test_len = len(testset)//num_task
            for i in range(num_task):
                """trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=False, transform=transform_train)
        
                testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=False, transform=transform_test)
                trainset.data, testset.data = permute([trainset.data, testset.data])
                trainloader_splits.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))"""

                
                trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len] = permute([trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len]])   
                trainloader_splits.append(torch.utils.data.DataLoader(Subset(trainset, range(i*train_len,(i+1)*train_len)), batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(Subset(testset, range(i*test_len,(i+1)*test_len)), batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))
                


    if dataset == "cifar10":
        image_size = 224
        num_classes = 10
        patch_size = 16
        dim = int(image_size/patch_size)**2 + 1
        code_dim = 1
        
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform_train)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=False, transform=transform_test)
        
        if task_type == 'normal':

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)


        if task_type == 'split':
            class_li = list(range(num_classes))
            np.random.shuffle(class_li)

            trainset_splits = split_dataset(trainset, n_splits=num_task, type = 'train', dataset = dataset, class_li = class_li)
            trainloader_splits = []
            for i, task in enumerate(trainset_splits):
                trainloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
            
            testset_splits = split_dataset(testset, n_splits=num_task, type = 'test', dataset = dataset, class_li = class_li)
            testloader_splits = []
            for i, task in enumerate(testset_splits):
                testloader_splits.append(torch.utils.data.DataLoader(task, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


        if task_type == 'permute':
            trainloader_splits = []
            testloader_splits = []
            train_len = len(trainset)//num_task
            test_len = len(testset)//num_task
            for i in range(num_task):
                trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len] = permute([trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len]])   
                trainloader_splits.append(torch.utils.data.DataLoader(Subset(trainset, range(i*train_len,(i+1)*train_len)), batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(Subset(testset, range(i*test_len,(i+1)*test_len)), batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


    if dataset == "cifar100":
        image_size = 224
        num_classes = 100
        patch_size = 16
        dim = int(image_size/patch_size)**2 + 1
        code_dim = 1

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=False, transform=transform_train)
        
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=False, transform=transform_test)
        
        if task_type == 'normal':

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)


        if task_type == 'split':
            class_li = list(range(num_classes))
            np.random.shuffle(class_li)

            trainset_splits = split_dataset(trainset, n_splits=num_task, type = 'train', dataset = dataset, class_li=class_li)
            trainloader_splits = []
            for i, task in enumerate(trainset_splits):
                trainloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                
            testset_splits = split_dataset(testset, n_splits=num_task, type = 'test', dataset = dataset, class_li=class_li)
            testloader_splits = []
            for i, task in enumerate(testset_splits):
                testloader_splits.append(torch.utils.data.DataLoader(task, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))

        if task_type == 'permute':
            trainloader_splits = []
            testloader_splits = []
            train_len = len(trainset)//num_task
            test_len = len(testset)//num_task
            for i in range(num_task):
                trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len] = permute([trainset.data[i*train_len:(i+1)*train_len], testset.data[i*test_len:(i+1)*test_len]])   
                trainloader_splits.append(torch.utils.data.DataLoader(Subset(trainset, range(i*train_len,(i+1)*train_len)), batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(Subset(testset, range(i*test_len,(i+1)*test_len)), batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


    if dataset == "imagenet-100":
        image_size = 224
        num_classes = 100
        patch_size = 16
        dim = int(image_size/patch_size)**2 + 1

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = datasets.ImageFolder(root='./data/imagenet-100/train',
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

        testset = datasets.ImageFolder(root='./data/imagenet-100/val',
                                            transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True)

        if task_type == 'normal':

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)

        if task_type == 'split':
            trainset_splits = split_dataset(trainset, n_splits=num_task, type = 'train', dataset = dataset)
            trainloader_splits = []
            for i, task in enumerate(trainset_splits):
                trainloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                
            testset_splits = split_dataset(testset, n_splits=num_task, type = 'test', dataset = dataset)
            testloader_splits = []
            for i, task in enumerate(testset_splits):
                testloader_splits.append(torch.utils.data.DataLoader(task, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))

        if task_type == 'permute':
            trainloader_splits = []
            testloader_splits = []
            for i in range(num_task):
                permute_trainset = datasets.ImageFolder(root='./data/imagenet-100/train',transform=transform_train)
                permute_testset = datasets.ImageFolder(root='./data/imagenet-100/val',transform=transform_test)
                permute_trainset.imgs, permute_testset.imgs = permute([permute_trainset.imgs, permute_testset.imgs], i)
                trainloader_splits.append(torch.utils.data.DataLoader(permute_trainset, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(permute_testset, batch_size=batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


    if dataset == "tin":
        image_size = 224
        num_classes = 200
        patch_size = 16 
        dim = int(image_size/patch_size)**2 + 1
        code_dim = 1
        
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',transform=transform_train)
        testset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',transform=transform_test)

        if task_type == 'normal':

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)


        if task_type == 'split':
            class_li = list(range(num_classes))
            np.random.shuffle(class_li)

            trainset_splits = split_dataset(trainset, n_splits=num_task, type = 'train', dataset = dataset, class_li=class_li)
            trainloader_splits = []
            #subset_indices = range(5000)
            for i, task in enumerate(trainset_splits):
                #task = Subset(task, subset_indices)
                trainloader_splits.append(torch.utils.data.DataLoader(task, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))

            testset_splits = split_dataset(testset, n_splits=num_task, type = 'test', dataset = dataset, class_li=class_li)
            testloader_splits = []
            for i, task in enumerate(testset_splits):
                #task = Subset(task, subset_indices)
                testloader_splits.append(torch.utils.data.DataLoader(task, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


        if task_type == 'permute':
            num = 10
            trainloader_splits = []
            testloader_splits = []
            for i in range(num):
                permute_trainset = torchvision.datasets.tiny_imagenet_dataset(root='./data', train=True, download=False, transform=transform_train)
                permute_testset = torchvision.datasets.tiny_imagenet_dataset(root='./data', train=False, download=False, transform=transform_test)
                permute_trainset.data, permute_testset.data = permute([permute_trainset.data, permute_testset.data], i)
                trainloader_splits.append(torch.utils.data.DataLoader(permute_trainset, batch_size=batch_size,
                                                        shuffle=True, num_workers=8, pin_memory=True))
                testloader_splits.append(torch.utils.data.DataLoader(permute_testset, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True))


    if dataset == "imagenet":
        image_size = 224
        num_classes = 1000
        patch_size = 16
        dim = int(image_size/patch_size)**2 + 1

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)), # resize the image to 256x256
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageNet(root='./data', split='train',
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0, pin_memory=True)

        testset = torchvision.datasets.ImageNet(root='./data', split='val',
                                            transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=0, pin_memory=True)


if modality == 'vision': 
    # vit paramereters
    if model_size == 'tiny':
        hidden_dim = 384
        net = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224") #torch.load('google-vit-base-patch16-224.pth')  #

    elif model_size == 'small':
        hidden_dim = 384
        net = ViTForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224") #torch.load('google-vit-base-patch16-224.pth')  #

    elif model_size == 'base':
        hidden_dim = 768
        net = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224") #torch.load('google-vit-base-patch16-224.pth')  #

    elif model_size == 'large':
        hidden_dim = 1024
        net = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224") #torch.load('google-vit-large-patch16-224.pth') #
    
    elif model_size == 'dino':
        hidden_dim = 384
        net = ViTForImageClassification.from_pretrained("facebook/dino-vits16") #torch.load('google-vit-base-patch16-224.pth')  #
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    net = BertModel.from_pretrained("bert-base-uncased")
    print(net)

net = net.cuda()

for i in range(len(list(net.vit.encoder.layer))):
    net.vit.encoder.layer[i].attention = ModularNN(n_k,num_expert,net.vit.encoder.layer[i].attention).cuda()

total_params = sum(p.numel() for p in net.parameters())
print(f'number of parameters - {total_params}')

# extract the embedding layer and set it untrainable
embedding_layer = copy.deepcopy(list(list(net.children())[0].children())[0])
for name, param in embedding_layer.named_parameters():
    param.requires_grad = False

# Create a list to store the autoencoders
if (dataset == 'cifar10') or (dataset == 'mnist'):
    autoencoders = nn.ModuleList([nn.DataParallel(Autoencoder(197,code_dim).cuda()) for i in range(num_task)])
else:
    autoencoders = nn.ModuleList([nn.DataParallel(Autoencoder(dim*hidden_dim,code_dim).cuda()) for i in range(num_task)])

ae_params = sum(p.numel() for p in autoencoders[0].parameters())
print(f"Total parameters of an AE: {ae_params}")



if mode == 'train' and retrain:
    PATH = f'./lora_{task_type}.pth'
    net.load_state_dict(torch.load(PATH))

if mode == "train":

    criterion = nn.CrossEntropyLoss()
    vae_criterion = nn.MSELoss()
    loss_mmd = MMD_loss()
    kl_loss = nn.KLDivLoss(log_target=True)

    #if not retrain:
    #    scheduler = CosineLRScheduler(optimizer, t_initial=epochs, warmup_t=5, warmup_lr_init=1e-5, lr_min = 1e-6, warmup_prefix=True) # do not define the min lr here
    
    # multi-GPUs/data parallel
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        embedding_layer = nn.DataParallel(embedding_layer)

    # Define a dictionary to store the training and testing accuracy, loss, and learning rate
    if task_type != "normal":
        history = {
            'train_acc': [],
            'train_loss': [],
            'lr_list': []
        }

        for i in range(num_task):
            history[f'test_acc_{i}'] = []
            history[f'test_loss_{i}'] = []
            history[f'ae_loss_{i}'] = []

    else:
        history = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': [],
            'lr_list': []
        }

    gateToExperts = {key: -1 for key in range(num_task)}
 
    start_time = time.time()
    aux_loss = 0
    memory_replay = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0     
        
        # AE is trained only once whenever a new task comes
        if epoch % (epochs//num_task) == 0:   
            task_order = (epoch*num_task)//epochs
            trainloader = trainloader_splits[task_order]

            # If it is larger than a threshold, then initialize a paired AE and adapter.

            # train AE
            # set only the gate AE as trainable
            for name, param in autoencoders.named_parameters():
                param.requires_grad = False
            for name, param in autoencoders[task_order].named_parameters():
                param.requires_grad = True
            optimizer_vae = optim.AdamW(filter(lambda p: p.requires_grad, autoencoders[task_order].parameters()), lr=vae_lr, betas=[0.9,0.999], weight_decay=0.05) # no weight decay in case of Adam
        

            if num_task != 1:
                # training AE models
                progress_bar = tqdm(total = vae_epochs)
                for e in range(vae_epochs): 
                    progress_bar.update(1)
                    ae_loss = 0.0

                    
                    for i,data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                
                        if inputs.shape[0] < batch_size:
                            continue
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        # zero the parameter gradients
                        optimizer_vae.zero_grad()

                        if (dataset == 'cifar10') or (dataset == 'mnist'):
                            input_to_ae = torch.sigmoid(torch.mean(embedding_layer(inputs), dim = 2))
                        else:
                            input_to_ae = torch.sigmoid(embedding_layer(inputs).flatten(start_dim = 1))
                        outputs = autoencoders[task_order](input_to_ae)
                        loss = vae_criterion(outputs, input_to_ae)
                        loss.backward()
                        optimizer_vae.step()

                        # print AE training loss
                        ae_loss += loss.cpu().item()

                    history[f'ae_loss_{task_order}'].append(ae_loss)

                progress_bar.close()
                print(f'[{epoch + 1}][AE: {task_order}] Training AE loss {loss.cpu().item():.3f}')

            """
            # memory replay based on herding
            all_data = []
            all_label = []
            for batch in trainloader:
                data, labels = batch
                all_data.append(data)
                all_label.append(labels)
            all_data = torch.cat(all_data).cuda()
            all_label = torch.cat(all_label).cuda()

            if (dataset == 'cifar10') or (dataset == 'mnist'):
                input_to_ae = torch.sigmoid(torch.mean(embedding_layer(all_data), dim = 2))
            else:
                input_to_ae = torch.sigmoid(embedding_layer(all_data).flatten(start_dim = 1))
               
            herd = input_to_ae
            # herding for the memory update
            herd = herd.flatten(start_dim=1)
            class_center = torch.mean(herd, dim=0)
            distances = torch.norm(herd - class_center, dim=1)
            sorted_indices = torch.argsort(distances)
            # Get the first several samples from herd based on sorted indices
            memory_replay.append((all_data[sorted_indices[:herd_size]],all_label[sorted_indices[:herd_size]]))
            "
            if (dataset == 'cifar10') or (dataset == 'mnist'):
                input_to_ae = torch.sigmoid(torch.mean(embedding_layer(all_data), dim = 2)) #[sorted_indices[:herd_size]]
            else:
                input_to_ae = torch.sigmoid(embedding_layer(all_data).flatten(start_dim = 1))
            
            # compute the gate of the most similar old task
            if task_order > 0:
                logits = torch.stack([vae_criterion(autoencoders[i](input_to_ae),input_to_ae) for i in range(task_order)])
                old_gate = gateToExperts[torch.argmin(logits).cpu().item()]
                print(torch.min(logits))
            # the first task, no previous tasks avaliable
            else:
            """
            logits = torch.tensor(1).cuda() # epsilon
           

            # update the old gate adapter for the new task
            if task_order >= limem or torch.min(logits).cpu().item() < epsilon:
                
                if torch.min(logits).cpu().item() < epsilon:
                    print(f"Recurring task and adapter {old_gate} is reused")
                else:
                    print(f"Capacity is reached and adapter {old_gate} is reused")

                # prepare dict for the memory replay
                transformed_dict = defaultdict(list)
                # Iterate over the original dictionary and append keys to corresponding values
                for key, value in gateToExperts.items():
                    transformed_dict[value].append(key)
                # Convert defaultdict back to a regular dictionary
                expertsToGate = dict(transformed_dict)

                
                # copy the old adapter to the new one (train on the old adapter with replay memory and new task data)
                for i in range(len(list(net.module.vit.encoder.layer))):
                    for key in net.module.vit.encoder.layer[i].attention.moe.experts[task_order].state_dict():
                        net.module.vit.encoder.layer[i].attention.moe.experts[task_order].state_dict()[key].data.copy_(net.module.vit.encoder.layer[i].attention.moe.experts[old_gate].state_dict()[key])
                
                
                # set only the last LoRA as trainable
                for name, param in net.named_parameters():
                    param.requires_grad = False
                
                for i in range(len(list(net.module.vit.encoder.layer))):
                    for name, param in net.module.vit.encoder.layer[i].attention.moe.experts[task_order].named_parameters():
                        if 'lora' in name:
                            param.requires_grad = True

            else:
                # set only the last LoRA as trainable
                for name, param in net.named_parameters():
                    param.requires_grad = False
                    
                for i in range(len(list(net.module.vit.encoder.layer))):
                    for name, param in net.module.vit.encoder.layer[i].attention.moe.experts[task_order].named_parameters():
                        if 'lora' in name:
                            param.requires_grad = True


            # when the capacity is full
            if task_order >= limem or torch.min(logits).cpu().item() < epsilon:
                gate = old_gate
                # ground truth for old tasks
                #net(memory_replay[old_gate]).logits.detach()
                # assign an expert to the task
            else:
                # assign an expert to the task
                gateToExperts[task_order] = task_order

            gate = task_order    
            # Count the number of trainable parameters
            lora_params = sum(p.numel() for p in net.module.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {lora_params}")

            # for each new task, we need to recreate the LoRA optimizer since different LoRAs are activated
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=[0.9,0.999], weight_decay=0.05) # no weight decay in case of Adam


        # Train the LoRA
        progress_bar = tqdm(total=len(trainloader))
        for i, data in enumerate(trainloader, 0):
            
            #if i > 0:
            #    break
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if inputs.shape[0] < batch_size:
                continue

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            """# forward + backward + optimize
            if task_order >= limem or torch.min(logits).cpu().item() < epsilon:
                outputs_tuple = net(inputs)
                outputs = outputs_tuple.logits
                new_loss = criterion(outputs, labels)
                
                # memory replay for all the old tasks corresponding to the old gate
                old_loss = torch.zeros(1).cuda()
                for e in expertsToGate[old_gate]:
                    gate = task_order
                    old_outputs_tuple = net(memory_replay[e][0])
                    old_outputs = old_outputs_tuple.logits
                    
                    old_logits = memory_replay[e][1]
                    old_loss += criterion(old_outputs, old_logits)

                gate = task_order
                alpha = 0.5
                # used KL loss with log max and deactivate the weight sharing between old and new tasks
                loss = new_loss * alpha + old_loss * (1-alpha)
            else:"""
            outputs_tuple = net(inputs)
            outputs = outputs_tuple.logits
            loss = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            # training acc
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            loss.backward()
            optimizer.step()
            
            # print training loss
            running_loss += loss.cpu().item()
            progress_bar.update(1)
            
        progress_bar.close()

        train_acc = 100*correct/total
        lr_step = optimizer.param_groups[0]['lr']
        history['train_loss'].append(running_loss)
        history['train_acc'].append(train_acc)
        # Record the current learning rate
        history['lr_list'].append(lr_step)
        
        # update the old gate adapter for the new task
        if task_order >= limem or torch.min(logits).cpu().item() < epsilon:

            gateToExperts[task_order] = old_gate

            for i in range(len(list(net.module.vit.encoder.layer))):
                for key in net.module.vit.encoder.layer[i].attention.moe.experts[gate].state_dict():
                    net.module.vit.encoder.layer[i].attention.moe.experts[old_gate].state_dict()[key].data.copy_(net.module.vit.encoder.layer[i].attention.moe.experts[gate].state_dict()[key])

        # evaluate every epoch
        # continual learning tasks acc
        if task_type != "normal":
            test_acc_li = []
            net.eval()
            embedding_layer.eval()
            for ae in autoencoders:
                ae.eval()
            with torch.no_grad():  
                for s, testloader in enumerate(testloader_splits):
        
                    test_loss = 0.0
                    correct = 0
                    total = 0
                    
                    # Accumulate data batches into a list
                    all_data = []
                    for batch in testloader:
                        data, labels = batch
                        all_data.append(data)
                    all_data = torch.cat(all_data)

                    # infer the gate based on the input
                    if (dataset == 'cifar10') or (dataset == 'mnist'):
                        input_to_ae = torch.sigmoid(torch.mean(embedding_layer(all_data), dim = 2))
                    else:
                        input_to_ae = torch.sigmoid(embedding_layer(all_data).flatten(start_dim = 1))
                    logits = torch.stack([vae_criterion(autoencoders[i](input_to_ae),input_to_ae) for i in range(num_task)])
                    gate_eval = gateToExperts[torch.argmin(logits).item()]
                    print(gate_eval)

                    for j, data in enumerate(testloader):
                        
                        correct_it = 0
                        total_it = 0
                        inputs, labels = data

                        if inputs.shape[0] < test_batch_size:
                            continue

                        inputs = inputs.cuda()
                        labels = labels.cuda()

                        outputs_tuple = net(inputs)

                        outputs = outputs_tuple.logits
                        
                        loss = criterion(outputs, labels)
                        test_loss += loss.cpu().item()
                        
                        # test acc
                        _, predictions = torch.max(outputs, 1)
                    
                        total += labels.size(0)
                    
                        correct += (predictions == labels).sum().item()
                        total_it += labels.size(0)
                        correct_it += (predictions == labels).sum().item()
                        test_acc_it = 100*correct_it/total_it
                    
                    test_acc = 100*correct/total
                    test_acc_li.append(test_acc)
                    history[f'test_acc_{s}'].append(test_acc)

                print(f'[{epoch + 1}][task: {task_order}] loss: {running_loss:.3f} training acc: {train_acc:.3f} test acc: {test_acc_li} lr: {lr_step:.6f}')


        # normal training task acc
        else:
            test_loss = 0.0
            correct = 0
            total = 0
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(testloader):
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    
                    outputs_tuple = net(inputs)

                    outputs = outputs_tuple.logits
                
                    loss = criterion(outputs, labels)
                    test_loss += loss.cpu().item()
                    
                    # test acc
                    _, predictions = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                
                test_acc = 100*correct/total
                
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            print(f'[{epoch + 1}] loss: {running_loss:.3f} training acc: {train_acc:.3f} test acc: {test_acc:.3f} lr: {lr_step:.6f}')


    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))

    # save the trained model weights
    if limem < num_task:
        PATH = f'lora_{task_type}_{dataset}_{num_task}_{model_size}_{limem}_{herd_size*num_task/num_classes}_{seed}.pth'
    else:
        PATH = f'lora_{task_type}_{dataset}_{num_task}_{model_size}_{seed}.pth'
    torch.save(net.module.state_dict(), PATH)
    
    # metrics
    # acc curve
    print(f'save history file to {task_type}_{dataset}_{num_task}_{model_size}_{seed}')
    if limem < num_task:
        torch.save(history, f'{task_type}_{dataset}_{num_task}_{model_size}_{limem}_{herd_size*num_task/num_classes}_{seed}.pt')
    else:
        torch.save(history, f'{task_type}_{dataset}_{num_task}_{model_size}_{seed}.pt')

    # average acc
    acc = 0
    forget = 0
    for i in range(num_task):
        acc += np.array((history[f'test_acc_{i}']))
        forget += np.array((history[f'test_acc_{i}'][(i+1)*len(history[f'test_acc_{i}'])//num_task-1])) - np.array((history[f'test_acc_{i}']))[-1]
    
    avg_acc = acc[-1] / num_task #np.max(acc/num_task)
    print(f"Average accuracy: {avg_acc}")

    # average forgetting rate
    avg_forget = forget / num_task
    print(f"Average forgetting rate: {avg_forget}")

    # memory footprint
    print(f"Parameters: LoRA {lora_params}, AE {ae_params}, Model {total_params}")


if mode == "eval":
    PATH = f'./lora.pth'
    net.load_state_dict(torch.load(PATH))
    
    # Evaluation of average test acc
    # evaluate every epoch for early stopping 
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for j, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            try:
                outputs_tuple = net(inputs)
            except:
                continue

            outputs = outputs_tuple.logits
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        
        print(f'Accuracy on the test images: {100*correct/total:.2f} %')
