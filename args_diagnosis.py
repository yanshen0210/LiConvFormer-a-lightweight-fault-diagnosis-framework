#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import numpy as np
import logging

from utils.train_val_test import train_val_test
from datasets.XJTU_gearbox import dataset_save as XJTU_gearbox
from datasets.XJTU_spurgear import dataset_save as XJTU_spurgear
from datasets.OU_bearing import dataset_save as OU_bearing

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='Liconvformer', help='the name of the model', choices=[
        'Liconvformer', 'CLFormer', 'convoformer_v1_small', 'mcswint',
         'MobileNet', 'MobileNetV2', 'ResNet18', 'MSResNet'])
    parser.add_argument('--save_dataset', type=bool, default=True, help='whether saving the dataset')
    parser.add_argument('--normalize_type', type=str, default='0-1', help='data normalization methods',
                        choices=['0-1', '-1-1', 'mean-std'])
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for each batch')

    # dataset parameters
    parser.add_argument('--dataset_name', type=str, default='XJTU_gearbox', help='the name of the dataset',
                        choices=['XJTU_gearbox', 'XJTU_spurgear', 'OU_bearing'])
    parser.add_argument('--sigma', type=int, default=0.0, help='the level of noise under noise task')

    # optimization information
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='the para of lr scheduler')
    parser.add_argument('--min_lr', type=int, default=1e-5, help='the para of lr scheduler')
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')

    # saving results
    parser.add_argument('--operation_num', type=int, default=5,
                        help='the repeat operation of model. If XJTU_spurgear, set 10; otherwise, set 5')
    parser.add_argument('--only_test', type=bool, default=False, help='loading the trained model if only test')

    args = parser.parse_args()
    return args


args = parse_args()

if args.save_dataset:
    if args.dataset_name == 'XJTU_gearbox':
        XJTU_gearbox(args)
    elif args.dataset_name == 'XJTU_spurgear':
        XJTU_spurgear(args)
    elif args.dataset_name == 'OU_bearing':
        OU_bearing(args)

else:
    # create the result dir
    save_dir = os.path.join('./results/{}'.format(args.dataset_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, args.model_name + '.log'))

    # save the args
    logging.info("\n")
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    logging.info('{}'.format(time))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    # model operation
    Accuracy = []
    J = []
    operation = train_val_test(args)
    for i in range(args.operation_num):
        if args.only_test == 0:
            operation.setup(i)
            operation.train_val(i)
        else:
            operation.setup(i)
        acc, j = operation.test(i)
        Accuracy.append(acc)
        J.append(j)

        if i == 4 or i == 9:
            Accuracy = np.array(Accuracy)*100
            Accuracy_mean = Accuracy.mean()
            Accuracy_var = Accuracy.var()
            Accuracy_max = Accuracy.max()
            Accuracy_min = Accuracy.min()
            J = np.array(J)
            J_mean = J.mean()
            J_var = J.var()
            J_max = J.max()
            J_min = J.min()

            Accuracy_list = ', '.join(['{:.2f}'.format(acc) for acc in Accuracy])
            J_list = ', '.join(['{:.2f}'.format(j) for j in J])
            logging.info('All acc: {}, \nMean acc: {:.2f}, Var acc {:.2f}, Max acc {:.2f}, Min acc {:.2f}'.format(
                Accuracy_list, Accuracy_mean, Accuracy_var, Accuracy_max, Accuracy_min))
            logging.info('All J: {}, \nMean J: {:.2f}, Var J {:.2f}, Max J {:.2f}, Min J {:.2f}\n'.format(
                J_list, J_mean, J_var, J_max, J_min))
            Accuracy = []
            J = []

