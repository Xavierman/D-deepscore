"""
This file is the evaluation score for the source task on the target task.
"""
import os
import sys
import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import svd
import time
from tqdm import tqdm

t1 = time.perf_counter()


# function to perform dimensionality reduction using SVD
def apply_SVD(X):
    U, _, _ = svd(X.T, full_matrices=False)  # Perform SVD on the transpose of X
    X_reduced = np.dot(X, U[:, :])
    return X_reduced


def getCov(X):
    X_mean = X
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
    return cov


# function to compute difference between distributions using SVD and Euclidean distance
def evaluate(f, Z):
    Z = np.argmax(Z, axis=1)
    alphabetZ = list(set(Z))
    g = np.zeros_like(f)
    for z in alphabetZ:
        Ef_z = np.mean(f[Z == z, :], axis=0)
        g[Z == z] = Ef_z

    f = apply_SVD(f)
    f = getCov(f)
    g = apply_SVD(g)
    g = getCov(g)

    score = np.trace(np.dot(np.linalg.pinv(f, rcond=1e-10), g))
    return score


def split_samples(X, split=0.9):
    train_indices = np.random.choice(len(X), int(len(X) * split), replace=False)
    test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
    X_train = X[train_indices]
    X_test = X[test_indices]
    return X_train, X_test, train_indices, test_indices  # Y_train,  Y_test,


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# load data
data_dir = "./dataset/NEU"
trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
testset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)

concat_dataset = torch.utils.data.ConcatDataset([trainset, testset])
print('Dataset_num:{}'.format(len(concat_dataset)))
trainloader = torch.utils.data.DataLoader(concat_dataset, batch_size=100, shuffle=False, num_workers=0)

# obtain image data and label data
y_train_batches = []
trainloader_bar = tqdm(trainloader, file=sys.stdout)
for _, y_batch in trainloader_bar:
    y_train_batches.append(y_batch)
y_train = torch.cat(y_train_batches, dim=0)

dataset_name = data_dir.split("/")[-1]
print('dataset:{}'.format(dataset_name))
pic_dir_out = f'./feature_{dataset_name}'
lossr = np.zeros((6, 2))
transferr = np.zeros((6, 2))
accr = np.zeros((6, 2, 2))

y_train_c = torch.nn.functional.one_hot(y_train)
y_train_c = y_train_c.numpy()
y_trn, y_tst, trn, tst = split_samples(y_train_c, split=0.8)

sort_mean_score = []
# 0.ImgN-pre.pth
load_pre_model = "./pre-model/0.ImgN-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean1 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean1)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean1}')

# 1.NEU-pre.pth
load_pre_model = "./pre-model/1.NEU-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean2 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean2)
print(f'D-deepscore({pre_model_name}):{evaluate_score}               Mean:{mean2}')

# 2.USB-pre.pth
load_pre_model = "./pre-model/2.USB-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean3 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean3)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean3}')

# 3.TAP-pre.pth
load_pre_model = "./pre-model/3.TAP-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean4 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean4)
print(f'D-deepscore({pre_model_name}):{evaluate_score}               Mean:{mean4}')

# 4.MaT-pre.pth
load_pre_model = "./pre-model/4.MaT-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean5 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean5)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean5}')

# 5.RSD-pre.pth
load_pre_model = "./pre-model/5.RSD-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean6 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean6)
print(f'D-deepscore({pre_model_name}):{evaluate_score}             Mean:{mean6}')

# 6.DAG-pre.pth
load_pre_model = "./pre-model/6.DAG-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean7 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean7)
print(f'D-deepscore({pre_model_name}):{evaluate_score}             Mean:{mean7}')

# 7.KyT-pre.pth
load_pre_model = "./pre-model/7.KyT-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean8 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean8)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean8}')

# 8.KTT-pre.pth
load_pre_model = "./pre-model/8.KTT-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean9 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean9)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean9}')

# 9.CSC-pre.pth
load_pre_model = "./pre-model/9.CSC-pre.pth"
pre_model_name = os.path.basename(load_pre_model)
# layer3-conv4
evaluate_score = []
for i in range(2):
    file_name = os.path.join(pic_dir_out, f'feature_c4_{i}_output_({pre_model_name}).h5')
    f = h5py.File(file_name, 'r')
    resnet18_train_output = f[f'feature_c4_{i}_output_({pre_model_name})'][:]
    f.close()
    resnet18_train_output_squeeze = np.squeeze(resnet18_train_output)
    score = evaluate(resnet18_train_output_squeeze[trn], y_trn)
    evaluate_score.append(score)
mean10 = sum(evaluate_score) / len(evaluate_score)
sort_mean_score.append(mean10)
print(f'D-deepscore({pre_model_name}):{evaluate_score}              Mean:{mean10}')

# sort
indexed_scores = [(index, score) for index, score in enumerate(sort_mean_score)]
sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
for rank, (index, score) in enumerate(sorted_scores, 1):
    print(f"Initial Position: {index + 1},   Data value: {score},   Sort: {rank}")
print('\ntime:{:.2f}s'.format(time.perf_counter() - t1))
