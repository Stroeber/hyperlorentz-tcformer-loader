import math
import torch
import torch.utils.data as Data
from scipy import io
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class MAMEMDataLoaderFM:
    def __init__(self, subject, ratio=8, data_path='./data/MAMEM/', bs=64, model=None, dev='cpu', finetune=False, patch_length=None):
        self.finetune = finetune
        self.subject = subject
        self.ratio = ratio
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 125
        self.patch_length = patch_length

        self.trainloader = None
        self.validloader = None
        self.testloader = None

    def pad_last_dim_to_multiple_of_200(self, tensor, pad_value=0):
        length = tensor.shape[-1]
        remainder = length % 200
        if remainder == 0:
            return tensor
        pad_amount = 200 - remainder
        # pad format: (pad_left, pad_right)
        return F.pad(tensor, (0, pad_amount), value=pad_value)


    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, 'U' + f'{int(self.subject):03d}' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)


        # permutation = torch.randperm(len(tempdata))
        # train = permutation[:300]
        # test = permutation[300:400]
        # val = permutation[400:500]

        # x_train = tempdata[train]
        # y_train = templabel[train]

        # x_valid = tempdata[test]
        # y_valid = templabel[test]
        #
        # x_test = tempdata[val]
        # y_test = templabel[val]

        x_train = tempdata[:300]
        y_train = templabel[:300]

        x_valid = tempdata[300:400]
        y_valid = templabel[300:400]

        x_test = tempdata[400:500]
        y_test = templabel[400:500]

        # x_test = tempdata[300:400]
        # y_test = templabel[300:400]
        #
        # x_valid = tempdata[400:500]
        # y_valid = templabel[400:500]

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)


        # total_count = tempdata.size(0)
        # train_count = int(0.7 * total_count)
        # valid_count = int(0.2 * total_count)
        # test_count = total_count - train_count - valid_count
        # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        #     model_dataset, (train_count, valid_count, test_count)
        # )

        if self.finetune:
            self.train_subject = torch.full((300, 1, 1, 125), int(self.subject), device=self.dev)
            self.valid_subject = torch.full((100, 1, 1, 125), int(self.subject), device=self.dev)
            self.test_subject = torch.full((100, 1, 1, 125), int(self.subject), device=self.dev)
            train_dataset = Data.TensorDataset(self.x_train, self.train_subject, self.y_train)
            valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subject, self.y_valid)
            test_dataset = Data.TensorDataset(self.x_test, self.test_subject, self.y_test)

        elif self.patch_length is not None :
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
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], num_patches, self.patch_length)
            self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.x_valid.shape[1], num_patches, self.patch_length)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], num_patches, self.patch_length)
            # print(f"Patched the training data to {self.x_train.shape}")
            # print(f"Patched the validation data to {self.x_valid.shape}")
            # print(f"Patched the test data to {self.x_test.shape}")
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


class MAMEMDataLoader_fixed:
    def __init__(self, subject, ratio=8, data_path='./data/MAMEM/', bs=64, model=None, dev='cpu'):
        self.subject = subject
        self.ratio = ratio
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 125

        self.trainloader = None
        self.validloader = None
        self.testloader = None

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, 'U' + f'{int(self.subject):03d}' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)

        x = tempdata.numpy()
        y = templabel.numpy()

        # Split the data into train, validation, and test sets with random splits while maintaining the same sizes
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4, )
        x_valid, x_test, y_valid, y_test = train_test_split(
            x_test, y_test, test_size=0.5, )

        # Convert the arrays to PyTorch tensors
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train).long()
        x_valid = torch.Tensor(x_valid)
        y_valid = torch.Tensor(y_valid).long()
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test).long()

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.to(self.dev)

        if self.finetune:
            self.train_subject = torch.full((self.bs, 1, 128), self.subject, device=self.dev)
            self.valid_subject = torch.full((self.bs, 1, 128), self.subject, device=self.dev)
            self.test_subject  = torch.full((self.bs, 1, 128), self.subject, device=self.dev)
            train_dataset = Data.TensorDataset(self.x_train, self.train_subject, self.y_train)
            valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subject, self.y_valid)
            test_dataset = Data.TensorDataset(self.x_test, self.test_subject, self.y_test)
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
