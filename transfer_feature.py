"""
This file is for extracting features from pre-trained resnet layer3-conv_4.
"""
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import h5py
import os
import time
import numpy as np
from tqdm import tqdm

t1 = time.perf_counter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# load data
data_dir = "./dataset/NEU"

dataset_name = data_dir.split("/")[-1]
pic_dir_out = f'./feature_{dataset_name}'
os.makedirs(pic_dir_out, exist_ok=True)

trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
testset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)

concat_dataset = torch.utils.data.ConcatDataset([trainset, testset])
print('Dataset_num:{}'.format(len(concat_dataset)))

trainloader = torch.utils.data.DataLoader(concat_dataset, batch_size=100, shuffle=False, num_workers=0)

# load model
num_classes = 2
base_model = models.resnet18(num_classes=num_classes).to(device)
load_pre_model = "./pre-model/9.CSC-pre.pth"
state_dict = torch.load(load_pre_model)
base_model.load_state_dict(state_dict)
base_model.eval()

#
# # layer2-conv3
# with torch.no_grad():
#     for i, k in enumerate(np.linspace(0, 1, 2)):
#         pre_model_name = os.path.basename(load_pre_model)
#         torch.cuda.empty_cache()
#         layer2_name = str(int(k))
#         batch_outputs = []
#         dataloader_val = tqdm(trainloader, file=sys.stdout)
#         for images, labels in dataloader_val:
#             images = images.cuda()
#             for name, module in base_model._modules.items():
#                 images = module(images)
#                 if name == "layer1":
#                     break
#             for name, module in base_model._modules['layer2']._modules.items():
#                 images = module(images)
#                 if name == f'{layer2_name}':
#                     break
#             feature = nn.AvgPool2d((28, 28))
#             outputs = feature(images)
#
#             batch_outputs.extend(outputs.cpu().numpy())
#             dataloader_val.set_description(f'resnet_conv3_{layer2_name}')
#         batch_outputs = np.array(batch_outputs)
#
#         file_name = os.path.join(pic_dir_out, f'feature__c3_{i}_output_({pre_model_name}).h5')
#
#         f = h5py.File(file_name, 'w')
#         f.create_dataset(f'feature_c3_{i}_output_({pre_model_name})', data=batch_outputs)
#         f.close()
#
#         print(f'feature__c3_(0-1)_output_({pre_model_name}) is done!')


# layer3-conv4
with torch.no_grad():
    for i, k in enumerate(np.linspace(0, 1, 2)):
        pre_model_name = os.path.basename(load_pre_model)
        torch.cuda.empty_cache()
        layer3_name = str(int(k))
        batch_outputs = []
        dataloader_val = tqdm(trainloader, file=sys.stdout)
        for images, labels in dataloader_val:
            images = images.cuda()

            for name, module in base_model._modules.items():
                images = module(images)
                if name == "layer2":
                    break
            for name, module in base_model._modules['layer3']._modules.items():
                images = module(images)
                if name == f'{layer3_name}':
                    break
            feature = nn.AvgPool2d((14, 14))
            outputs = feature(images)

            batch_outputs.extend(outputs.cpu().numpy())
            dataloader_val.set_description(f'resnet_conv4_{layer3_name}')
        batch_outputs = np.array(batch_outputs)

        file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')

        f = h5py.File(file_name, 'w')
        f.create_dataset(f'feature_c4_{i}_output_({pre_model_name})', data=batch_outputs)
        f.close()

        print(f'feature_c4_output_({pre_model_name}) is done!')

# # layer4-conv5
# with torch.no_grad():
#     for i, k in enumerate(np.linspace(0, 1, 2)):
#         pre_model_name = os.path.basename(load_pre_model)
#         torch.cuda.empty_cache()
#         layer4_name = str(int(k))
#         batch_outputs = []
#         dataloader_val = tqdm(trainloader, file=sys.stdout)
#         for images, labels in dataloader_val:
#             images = images.cuda()
#             for name, module in base_model._modules.items():
#                 images = module(images)
#                 if name == "layer3":
#                     break
#             for name, module in base_model._modules['layer4']._modules.items():
#                 images = module(images)
#                 if name == f'{layer4_name}':
#                     break
#             feature = nn.AvgPool2d((7, 7))
#             outputs = feature(images)
#
#             batch_outputs.extend(outputs.cpu().numpy())
#             dataloader_val.set_description(f'resnet_conv5_{layer4_name}')
#         batch_outputs = np.array(batch_outputs)
#
#         file_name = os.path.join(pic_dir_out, f'feature_c5_{i}_output_({pre_model_name}).h5')
#
#         f = h5py.File(file_name, 'w')
#         f.create_dataset(f'feature_c5_{i}_output_({pre_model_name})', data=batch_outputs)
#         f.close()
#
#         print(f'feature_c5_(0-1)_output_({pre_model_name}) is done!')
#
# print('time:{:.2f}s'.format(time.perf_counter()-t1))
