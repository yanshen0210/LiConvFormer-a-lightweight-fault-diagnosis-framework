import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from scipy.io import savemat, loadmat

from datasets.data_pre import *


label = [i for i in range(0, 5)]

# load dataset
class dataset(Dataset):
    def __init__(self, list_data, transform):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        lab = self.labels[item]
        seq = self.transforms(seq)
        return seq, lab


# save dataset
def dataset_save(args):
    data1, lab1 = [], []
    data2, lab2 = [], []
    data3, lab3 = [], []
    data_dir = './data/v43hmbwxpm-2'
    fault_mode = os.listdir(os.path.join(data_dir))

    for i in tqdm(range(len(label))):  # the number of fault mode
        datasets = os.listdir(os.path.join(data_dir, fault_mode[i]))

        path1 = os.path.join(data_dir+'/'+fault_mode[i], datasets[0])
        data, lab = data_load(args, path1, label=label[i])
        data1 += data
        lab1 += lab

        path2 = os.path.join(data_dir + '/' + fault_mode[i], datasets[1])
        data, lab = data_load(args, path2, label=label[i])
        data2 += data
        lab2 += lab

        path3 = os.path.join(data_dir + '/' + fault_mode[i], datasets[2])
        data, lab = data_load(args, path3, label=label[i])
        data3 += data
        lab3 += lab

    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data1, lab1, data2, lab2, data3, lab3]
    np.save('./data/save_dataset/' + args.dataset_name + '.npy', list_data)


# load data from the file
def data_load(args, root, label):
    data = []
    lab = []
    fl = loadmat(root)['Channel_1']
    fl = np.array(fl, dtype=np.float32).reshape(-1)
    fl = fl*100

    length = 1e5+1500*600  # all samples
    start, end = int(1e5), int(1e5+1024)
    while end <= length:
        # x = np.fft.fft(fl[start:end])
        # x = np.abs(x) / len(x)
        # x = x[range(int(x.shape[0] / 2))]
        x = fl[start:end].reshape(1, -1)
        data.append(x)
        lab.append(label)
        start += 1500
        end += 1500
    return data, lab


class OU_bearing(object):
    num_sensor = 1
    num_classes = 5

    # load dataset for operation
    def data_prepare(self, args, op_num):

        # load the datasets
        list_data = np.load('./data/save_dataset/' + args.dataset_name + '.npy', allow_pickle=True)
        train_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        val_pd = pd.DataFrame({"data": list_data[2], "label": list_data[3]})
        test_pd = pd.DataFrame({"data": list_data[4], "label": list_data[5]})

        # different operations using different datasets to train, val and test
        _, val_pd = train_test_split(val_pd, test_size=1/2,
                                     random_state=op_num, stratify=val_pd["label"])
        _, test_pd = train_test_split(test_pd, test_size=1/2,
                                      random_state=op_num, stratify=test_pd["label"])

        # the way of data preprocess
        train_preprocess = Compose([Normalize(args.normalize_type), Retype()])
        test_preprocess = Compose([RandomAddGaussian(args.sigma), RandomScale(args.sigma),
                                   Normalize(args.normalize_type), Retype()])

        train_dataset = dataset(list_data=train_pd, transform=train_preprocess)
        val_dataset = dataset(list_data=val_pd, transform=train_preprocess)
        test_dataset = dataset(list_data=test_pd, transform=test_preprocess)
        return train_dataset, val_dataset, test_dataset

