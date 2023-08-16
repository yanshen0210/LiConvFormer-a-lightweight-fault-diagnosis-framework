import time
import torch
import torch.nn as nn
import numpy as np
import os
from torch import optim
import logging
# from torchsummary import summary
from thop import profile, clever_format
# from torchstat import stat

import models
import datasets


class train_val_test(object):
    def __init__(self, args):
        self.args = args

    def setup(self, op_num):
        """
        Initialize the datasets, model, loss and optimizer
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # print('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            # print('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets, args.dataset_name)
        self.datasets = {}
        self.datasets['train'], self.datasets['val'], self.datasets['test'] = Dataset().data_prepare(args, op_num)
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=(args.batch_size if x == 'train' else 50),
                                                           shuffle=(True if x == 'train' else False),
                                                           drop_last=(True if x == 'test' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val', 'test']}

        # Define the model
        self.num_sensor = Dataset.num_sensor
        self.num_classes = Dataset.num_classes
        self.model = getattr(models, args.model_name)(args, in_channel=self.num_sensor, out_channel=self.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer and learning rate decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience,
                                                                 min_lr=args.min_lr, verbose=True)

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_val(self, op_num):
        """
        Training and validation process
        """
        args = self.args
        train_time = 0.0
        best_acc = 0
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        for epoch in range(args.epoch):
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for _, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Do the learning process, in val, do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                # logging the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)

                # calculate the training time
                if phase == 'train':
                    train_time += time.time()-epoch_start
                    logging.info(' ')
                    self.lr_scheduler.step(epoch_loss)
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                elif phase == 'val':
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)

                logging.info('Num-{}, Epoch: {}-{} Loss: {:.4f}, Acc: {:.4f}, Time {:.4f} sec'.format(
                    op_num, epoch, phase, epoch_loss, epoch_acc, time.time()-epoch_start))

            # save the model of the best val acc
            if epoch >= args.epoch//2 and epoch_acc >= best_acc and phase == 'val':
                    best_acc = epoch_acc
                    best_epoch = epoch
                    save_dir = os.path.join(
                        './trained_models/{}/{}'.format(args.dataset_name, args.model_name))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self.model.state_dict(), os.path.join('{}/{}.pth'.format(
                        save_dir, 'operation_' + str(op_num))))

        logging.info("\nTraining time {:.2f}, Best_epoch {}".format(train_time, best_epoch))

        if args.dataset_name == 'XJTU_gearbox':
            mode = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy']
            value_dict = {0: train_loss, 1: train_acc, 2: val_loss, 3: val_acc}
            for i in range(len(mode)):
                loss_acc = os.path.join('./results/XJTU_gearbox/{}'.format(mode[i]))
                if not os.path.exists(loss_acc):
                    os.makedirs(loss_acc)
                with open("{}/{}.txt".format(loss_acc, args.model_name+'_'+str(op_num)), 'w') as txt:
                    txt.write(str(value_dict.get(i)))

    def test(self, op_num):
        """
        Test process
        """
        args = self.args

        # loading the best trained model
        save_dir = os.path.join(
            './trained_models/{}/{}'.format(args.dataset_name, args.model_name))
        self.model.load_state_dict(torch.load('{}/{}.pth'.format(
                save_dir, 'operation_' + str(op_num))), strict=False)

        feature, label_pre, label_true = [], [], []
        acc = 0
        loss_all = 0.0
        self.model.eval()
        test_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['test']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)
            loss_all += loss_temp
            acc += correct
            label_pre.append(pred)
            label_true.append(labels)
            feature += [tensor.cpu().detach() for tensor in logits]

        sample_num = len(feature)
        loss = loss_all / sample_num
        acc = acc / sample_num
        test_time = time.time() - test_start

        Sb, Sw, J1 = intraclass_covariance(feature, torch.stack(label_true, 0).cpu(), sample_num, self.num_classes)
        logging.info('Num-{}, Test  Loss: {:.4f}, Acc: {:.4f}, Time {:.4f} sec'.format(
            op_num, loss, acc, test_time))
        logging.info('     Sb: {:.4f}, Sw: {:.4f}, J1: {:.4f} \n'.format(
            Sb, Sw, J1))

        # saving the labels of prediction and reality
        if args.dataset_name == 'XJTU_spurgear':
            label_pre = np.array(torch.stack(label_pre, 0).cpu()).ravel()
            label_true = np.array(torch.stack(label_true, 0).cpu()).ravel()
            feature = np.array(torch.stack(feature, 0).cpu())
            save_dir1 = os.path.join('./results/{}/Pre label/'.format(args.dataset_name))
            save_dir2 = os.path.join('./results/{}/True label/'.format(args.dataset_name))
            save_dir3 = os.path.join('./results/{}/Feature/'.format(args.dataset_name))
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)
            if not os.path.exists(save_dir3):
                os.makedirs(save_dir3)
            np.savetxt('{}/{}_{}_{}.txt'.format(save_dir1, args.model_name, args.sigma, op_num),
                       label_pre, fmt='%.0f', newline='\n')
            np.savetxt('{}/{}_{}_{}.txt'.format(save_dir2, args.model_name, args.sigma, op_num),
                       label_true, fmt='%.0f', newline='\n')
            np.savetxt('{}/{}_{}_{}.txt'.format(save_dir3, args.model_name, args.sigma, op_num),
                       feature, newline='\n')

        if op_num == args.operation_num-1:
            flops, params = profile(self.model, inputs=(torch.randn(1, self.num_sensor, 1024).to(self.device),))
            flops, params = clever_format([flops, params], "%.3f")
            logging.info('flops:{}, params:{}\n'.format(flops, params))

        return acc, J1


def intraclass_covariance(test_data, label, sum_n, classes):
    """
    test_data:输出特征
    label:真实标签
    sum_n: 总测试个数
    classes:总的类别数
    """
    # Nk = sum_n//classes
    mk = []
    x = [tensor.numpy() for tensor in test_data]
    x = np.stack(x, axis=0)
    y = label.numpy().reshape(sum_n)
    Sb, Sw = 0, 0
    for i in range(classes):
        Nk = len(x[y == i])
        cur_mean = np.sum(x[y == i], axis=0) / Nk
        mk.append(cur_mean)
    m = np.mean(x, axis=0)

    for j in range(classes):
        Nk = len(x[y == j])
        Sb += Nk * np.linalg.norm((mk[j] - m))
        x_class = x[y == j]
        for k in range(Nk):
            Sw += np.linalg.norm((x_class[k] - mk[j]))

    J1 = Sb / Sw
    return Sb, Sw, J1