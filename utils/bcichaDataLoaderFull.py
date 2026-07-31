import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
import os

from utils.pytorch_data import make_collate_fn_inter


class BCIchaDataLoaderFull:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, dev='cpu'):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 160

    def get_dataloader(self, interaug=False):
        x_train_list, y_train_list = [], []
        x_valid_list, y_valid_list = [], []
        x_test_list, y_test_list = [], []
        train_subjects_list, valid_subjects_list, test_subjects_list = [], [], []

        for i,sub in enumerate([2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26]):

            subject_id = np.array(i+1).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 160, axis=2)

            reshape_dim_train = (2880, 1, 1, 160)
            reshape_dim_valid = (960, 1, 1, 160)
            reshape_dim_test = (1600, 1, 1, 160)

            train = io.loadmat(os.path.join(self.data_path, f'Data_S{sub:02d}_Sess' + '.mat'))

            tempdata = torch.Tensor(train['x_test'])
            templabel = torch.Tensor(train['y_test']).view(-1)
            x_train = tempdata[:180]
            y_train = templabel[:180]

            x_valid = tempdata[180:240]
            y_valid = templabel[180:240]

            x_test = tempdata[240:340]
            y_test = templabel[240:340]

            x_train_list.append(x_train)
            x_valid_list.append(x_valid)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_valid_list.append(y_valid)
            y_test_list.append(y_test)
            train_subjects_list.append(np.repeat(reshaped_weights, 180, axis=0))
            valid_subjects_list.append(np.repeat(reshaped_weights, 60, axis=0))
            test_subjects_list.append(np.repeat(reshaped_weights, 100, axis=0))


        x_train = torch.cat(x_train_list, dim=0)
        x_valid = torch.cat(x_valid_list, dim=0)
        x_test = torch.cat(x_test_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)
        y_valid = torch.cat(y_valid_list, dim=0)
        y_test = torch.cat(y_test_list, dim=0)
        train_subjects = torch.Tensor(np.concatenate(train_subjects_list, axis=0))
        valid_subjects = torch.Tensor(np.concatenate(valid_subjects_list, axis=0))
        test_subjects = torch.Tensor(np.concatenate(test_subjects_list, axis=0))

        x_train = torch.reshape(x_train, (2880, 1, 56, 160))
        train_subjects = torch.reshape(train_subjects, reshape_dim_train)
        x_valid = torch.reshape(x_valid, (960, 1, 56, 160))
        valid_subjects = torch.reshape(valid_subjects, reshape_dim_valid)
        x_test = torch.reshape(x_test, (1600, 1, 56, 160))
        test_subjects = torch.reshape(test_subjects, reshape_dim_test)

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)
        self.train_subjects = train_subjects.to(self.dev)
        self.valid_subjects = valid_subjects.to(self.dev)
        self.test_subjects = test_subjects.to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.test_subjects, self.y_test)

        trainloader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=make_collate_fn_inter(interaug)
        )
        validloader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        testloader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return trainloader, validloader, testloader

from sklearn.model_selection import train_test_split

class BCIchaDataLoaderFull_fixed:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, dev='cpu'):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 160

    def get_dataloader(self):
        x_list, y_list, subjects_list = [], [], []

        for i, sub in enumerate([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]):

            subject_id = np.array(i+1).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 160, axis=2)

            train = io.loadmat(os.path.join(self.data_path, f'Data_S{sub:02d}_Sess' + '.mat'))

            tempdata = torch.Tensor(train['x_test'])
            templabel = torch.Tensor(train['y_test']).view(-1)

            x_list.append(tempdata)
            y_list.append(templabel)
            subjects_list.append(np.repeat(reshaped_weights, tempdata.shape[0], axis=0))

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        subjects = torch.Tensor(np.concatenate(subjects_list, axis=0))

        # Split the data into train, validation, and test sets with random splits while maintaining the same sizes
        x_train, x_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
            x, y, subjects, test_size=0.47, )
        x_valid, x_test, y_valid, y_test, subjects_valid, subjects_test = train_test_split(
            x_test, y_test, subjects_test, test_size=0.625, )


        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)
        self.train_subjects = subjects_train.to(self.dev)
        self.valid_subjects = subjects_valid.to(self.dev)
        self.test_subjects = subjects_test.to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.test_subjects, self.y_test)

        trainloader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        validloader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        testloader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return trainloader, validloader, testloader
