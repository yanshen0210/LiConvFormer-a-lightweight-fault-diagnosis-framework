import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.utils import shuffle

from datasets.data_pre import *


label = [i for i in range(0, 9)]

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
    data1 = []
    lab1 = []
    data_dir = './data/XJTU_Gearbox'
    fault_mode = os.listdir(os.path.join(data_dir))

    for i in tqdm(range(9)):  # the number of fault mode
        channel = os.listdir(os.path.join(data_dir, fault_mode[i]))  # two sensor channels

        path1 = os.path.join(data_dir+'/'+fault_mode[i], channel[0])
        path2 = os.path.join(data_dir + '/' + fault_mode[i], channel[1])
        data, lab = data_load(args, path1, path2, label=label[i])
        data1 += data
        lab1 += lab

    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data1, lab1]
    np.save('./data/save_dataset/' + args.dataset_name + '.npy', list_data)


# load data from the file
def data_load(args, root1, root2, label):
    data = []
    lab = []
    fl = pd.read_csv(root1, sep='\t',  header=None, skiprows=100)
    fl = fl.values
    fl = np.array(fl, dtype=np.float32).reshape(-1)
    f2 = pd.read_csv(root2, sep='\t',  header=None, skiprows=100)
    f2 = f2.values
    f2 = np.array(f2, dtype=np.float32).reshape(-1)

    length = 1500*1200  # all samples
    start, end = 0, 1024
    while end <= length:
        x1 = fl[start:end]
        x2 = f2[start:end]
        sample = np.concatenate((x1, x2), axis=0).reshape(2, -1)
        data.append(sample)
        lab.append(label)
        start += 1500
        end += 1500
    return data, lab


class XJTU_gearbox(object):
    num_sensor = 2
    num_classes = 9

    # load dataset for operation
    def data_prepare(self, args, op_num):

        # load the datasets
        list_data = np.load('./data/save_dataset/' + args.dataset_name + '.npy', allow_pickle=True)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        # different operations using different datasets to train, val and test
        train_pd, val_test_pd = train_test_split(data_pd, test_size=7/12,
                                                 random_state=op_num, stratify=data_pd["label"])
        val_pd, test_pd = train_test_split(val_test_pd, test_size=4/7,
                                           random_state=op_num, stratify=val_test_pd["label"])
        # test_pd = test_pd.sort_values('label')  # sorting the test set

        # the way of data preprocess
        train_preprocess = Compose([Normalize(args.normalize_type), Retype()])
        test_preprocess = Compose([RandomAddGaussian(args.sigma), RandomScale(args.sigma),
                                   Normalize(args.normalize_type), Retype()])

        train_dataset = dataset(list_data=train_pd, transform=train_preprocess)
        val_dataset = dataset(list_data=val_pd, transform=train_preprocess)
        test_dataset = dataset(list_data=test_pd, transform=test_preprocess)
        return train_dataset, val_dataset, test_dataset

