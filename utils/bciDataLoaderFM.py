import math
import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os
import torch.nn.functional as F


class BCIDataLoaderFM:
    def __init__(self, subject, ratio, data_path, bs, model=None, dev='cpu', finetune=False, patch_length=None):
        self.finetune = finetune
        self.subject = subject
        self.ratio = ratio
        self.data_path = data_path
        self.bs = bs
        self.dev = dev
        self.model = model
        self.timeseries_length = 438
        self.patch_length = patch_length
        self.num_patches = self.timeseries_length // self.patch_length

    def split_train_valid_set(self, x_train, y_train, ratio):
        s = y_train.argsort()
        x_train = x_train[s]
        y_train = y_train[s]

        cL = int(len(x_train) / 4)

        class1_x = x_train[0 * cL: 1 * cL]
        class2_x = x_train[1 * cL: 2 * cL]
        class3_x = x_train[2 * cL: 3 * cL]
        class4_x = x_train[3 * cL: 4 * cL]

        class1_y = y_train[0 * cL: 1 * cL]
        class2_y = y_train[1 * cL: 2 * cL]
        class3_y = y_train[2 * cL: 3 * cL]
        class4_y = y_train[3 * cL: 4 * cL]

        vL = int(len(class1_x) / ratio)

        x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
        y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

        x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
        y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

        return x_train, y_train, x_valid, y_valid

    def pad_last_dim_to_multiple_of_200(self, tensor, pad_value=0):
        length = tensor.shape[-1]
        remainder = length % 200
        if remainder == 0:
            return tensor
        pad_amount = 200 - remainder
        # pad format: (pad_left, pad_right)
        return F.pad(tensor, (0, pad_amount), value=pad_value)

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, f'BCIC_S{int(self.subject):02d}_T.mat'))
        test = io.loadmat(os.path.join(self.data_path, f'BCIC_S{int(self.subject):02d}_E.mat'))

        if self.model == 'matt':
            x_train = torch.Tensor(train['x_train']).unsqueeze(1)
            x_test = torch.Tensor(test['x_test']).unsqueeze(1)
        else:
            x_train = torch.Tensor(train['x_train'])
            x_test = torch.Tensor(test['x_test'])

        y_train = torch.Tensor(train['y_train']).view(-1)
        y_test = torch.Tensor(test['y_test']).view(-1)

        x_train, y_train, x_valid, y_valid = self.split_train_valid_set(x_train, y_train, ratio=self.ratio)

        if self.model == 'matt':
            self.x_train = x_train[:, :, :, 124:562].to(self.dev)
            self.x_valid = x_valid[:, :, :, 124:562].to(self.dev)
            self.x_test = x_test[:, :, :, 124:562].to(self.dev)
        elif self.patch_length is not None:
            # print(x_train.shape)
            # Padding zeros so the tensor is divisible by 200
            self.x_train = self.pad_last_dim_to_multiple_of_200(
                x_train[:, :, 124:562].to(self.dev))
            self.x_valid = self.pad_last_dim_to_multiple_of_200(
                x_valid[:, :, 124:562].to(self.dev))
            self.x_test = self.pad_last_dim_to_multiple_of_200(
                x_test[:, :, 124:562].to(self.dev))
            # print(f"Dimensions after padding is {self.x_train.shape}")
            num_patches = math.ceil(self.timeseries_length / self.patch_length)
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], num_patches,
                                                self.patch_length)
            self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.x_valid.shape[1], num_patches,
                                                self.patch_length)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], num_patches,
                                              self.patch_length)
            # print(f"Patched the training data to {self.x_train.shape}")
            # print(f"Patched the validation data to {self.x_valid.shape}")
            # print(f"Patched the test data to {self.x_test.shape}")


        else:
            self.x_train = x_train[:, :, 124:562].to(self.dev)
            self.x_valid = x_valid[:, :, 124:562].to(self.dev)
            self.x_test = x_test[:, :, 124:562].to(self.dev)
            # if self.patch_length is not None:
            #     print(x_train.shape)
            #     self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.num_patches, self.patch_length)
            #     self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.x_valid.shape[1], self.num_patches, self.patch_length)
            #     self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.num_patches, self.patch_length)
            #     print(f"Patched the training data to {self.x_train.shape}")
            #     print(f"Patched the validation data to {self.x_valid.shape}")
            #     print(f"Patched the test data to {self.x_test.shape}")

        self.y_train = y_train.long().to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        if self.finetune:
            train_subjects = torch.full((252, 1, 1, self.timeseries_length), self.subject)
            valid_subjects = torch.full((36, 1, 1, self.timeseries_length), self.subject)
            test_subjects = torch.full((288, 1, 1, self.timeseries_length), self.subject)

            self.train_subjects = train_subjects.to(self.dev)
            self.valid_subjects = valid_subjects.to(self.dev)
            self.test_subjects = test_subjects.to(self.dev)

            self.train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
            self.valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
            self.test_dataset = Data.TensorDataset(self.x_test, self.test_subjects, self.y_test)
        else:
            self.train_dataset = Data.TensorDataset(self.x_train, self.y_train)
            self.valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
            self.test_dataset = Data.TensorDataset(self.x_test, self.y_test)

        trainloader = Data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        validloader = Data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        testloader = Data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return trainloader, validloader, testloader



