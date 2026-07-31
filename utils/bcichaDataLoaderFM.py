import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
import os


class BCIchaDataLoaderFM:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, model=None, dev='cpu', finetune=None,
                 patch_length=200):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 160
        self.finetune = finetune
        self.patch_length = patch_length
        # Uncomment the following line if you want to use GPU
        # self.dev = torch.device("cuda")

    def pad_last_dim_to_multiple_of_200(self, tensor, pad_value=0):
        length = tensor.shape[-1]
        remainder = length % 200
        if remainder == 0:
            return tensor
        pad_amount = 200 - remainder
        # pad format: (pad_left, pad_right)
        return F.pad(tensor, (0, pad_amount), value=pad_value)

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, f'Data_S{int(self.subject):02d}_Sess' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)
        x_train = tempdata[:180]
        y_train = templabel[:180]

        x_valid = tempdata[180:240]
        y_valid = templabel[180:240]

        x_test = tempdata[240:340]
        y_test = templabel[240:340]

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        if self.finetune:

            ids = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
            lookup = {id_val: idx + 1 for idx, id_val in enumerate(ids)}

            train_subjects = torch.full((180, 1, 1, 160), lookup.get(self.subject))
            valid_subjects = torch.full((60, 1, 1, 160), lookup.get(self.subject))
            test_subjects = torch.full((100, 1, 1, 160), lookup.get(self.subject))

            self.train_subjects = train_subjects.to(self.dev)
            self.valid_subjects = valid_subjects.to(self.dev)
            self.test_subjects = test_subjects.to(self.dev)

            train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
            valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
            test_dataset = Data.TensorDataset(self.x_test, self.test_subjects, self.y_test)

        elif self.patch_length is not None:
            print(self.x_train.shape)
            # Padding zeros so the tensor is divisible by 200
            self.x_train = self.pad_last_dim_to_multiple_of_200(
                self.x_train.to(self.dev))
            self.x_valid = self.pad_last_dim_to_multiple_of_200(
                self.x_valid.to(self.dev))
            self.x_test = self.pad_last_dim_to_multiple_of_200(
                self.x_test.to(self.dev))
            # print(f"Dimensions after padding is {self.x_train.shape}")
            num_patches = math.ceil(self.timeseries_length / self.patch_length)
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], num_patches,
                                                self.patch_length)
            self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.x_valid.shape[1], num_patches,
                                                self.patch_length)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], num_patches,
                                              self.patch_length)
            print(f"Patched the training data to {self.x_train.shape}")
            print(f"Patched the validation data to {self.x_valid.shape}")
            print(f"Patched the test data to {self.x_test.shape}")
            train_dataset = Data.TensorDataset(self.x_train, self.y_train)
            valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
            test_dataset = Data.TensorDataset(self.x_test, self.y_test)
        else:
            train_dataset = Data.TensorDataset(self.x_train, self.y_train)
            valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
            test_dataset = Data.TensorDataset(self.x_test, self.y_test)

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


from sklearn.model_selection import train_test_split


class BCIchaDataLoader_fixed:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, model=None, dev='cpu'):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 160

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, f'Data_S{int(self.subject):02d}_Sess' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)

        # Split the data into train, validation, and test sets with random splits while maintaining the same sizes
        x_train, x_test, y_train, y_test = train_test_split(tempdata, templabel, test_size=0.47, )
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.625, )

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)

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
