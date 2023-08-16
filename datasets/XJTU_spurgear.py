import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.utils import shuffle

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
        label = self.labels[item]
        seq = self.transforms(seq)
        return seq, label


# save dataset
def dataset_save(args):
    data1 = []
    lab1 = []
    data2 = []
    lab2 = []
    data_dir1 = './data/XJTU_Spurgear/15Hz'
    data_dir2 = './data/XJTU_Spurgear/20Hz'
    operation = os.listdir(os.path.join(data_dir1))

    for i in tqdm(range(5)):  # the number of fault mode
        path1 = os.path.join(data_dir1, operation[i])  # the root of dataset1
        data, lab = data_load(args, path1, label=label[i])
        data1 += data
        lab1 += lab
        path2 = os.path.join(data_dir2, operation[i])  # the root of dataset2
        data, lab = data_load(args, path2, label=label[i])
        data2 += data
        lab2 += lab
    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data1, lab1, data2, lab2]
    np.save('./data/save_dataset/' + args.dataset_name + '.npy', list_data)


# load data from the file
def data_load(args, root, label):
    data = []
    lab = []
    fl = pd.read_csv(root, sep='\t',  header=None, skiprows=10000)
    fl = fl.values
    fl = np.array(fl, dtype=np.float32)
    length = 1500*600

    start, end = 0, 1024
    while end <= length:
        sample = []
        for j in range(12):
            x = fl[start:end, j+1]
            sample.append(x)
        sample = np.array(sample)
        data.append(sample)
        lab.append(label)
        start += 1500
        end += 1500
    return data, lab


class XJTU_spurgear(object):
    num_sensor = 12
    num_classes = 5

    # load dataset for operation
    def data_prepare(self, args, op_num):

        # load the datasets
        list_data = np.load('./data/save_dataset/' + args.dataset_name + '.npy',
                            allow_pickle=True)

        # the way of data preprocess
        train_preprocess = Compose([Normalize(args.normalize_type), Retype()])
        test_preprocess = Compose([RandomAddGaussian(args.sigma), RandomScale(args.sigma),
                                   Normalize(args.normalize_type), Retype()])
        data_pd1 = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        data_pd2 = pd.DataFrame({"data": list_data[2], "label": list_data[3]})

        if op_num < 5:
            train_pd, val_pd = train_test_split(data_pd1, test_size=1 / 3,
                                                random_state=op_num, stratify=data_pd1["label"])
            test_pd = data_pd2

        else:
            train_pd, val_pd = train_test_split(data_pd2, test_size=1 / 3,
                                                random_state=op_num, stratify=data_pd2["label"])
            test_pd = data_pd1
        # else:
        #     raise Exception("the operation_num is set wrong, it should be 10 in this dataset")

        train_dataset = dataset(list_data=train_pd, transform=train_preprocess)
        val_dataset = dataset(list_data=val_pd, transform=train_preprocess)
        test_dataset = dataset(list_data=test_pd, transform=test_preprocess)
        return train_dataset, val_dataset, test_dataset



