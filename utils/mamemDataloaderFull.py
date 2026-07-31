import torch
import torch.utils.data as Data
from scipy import io
import os
import numpy as np

from utils.pytorch_data import make_collate_fn_inter


class MAMEMDataLoaderFull:
    def __init__(self, subject, data_path='./data/MAMEM/', bs=64, dev='cpu'):
        self.subject = subject
        self.ratio = 8
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 125

        self.trainloader = None
        self.validloader = None
        self.testloader = None

    def get_dataloader(self, interaug=False):

        x_train_list, y_train_list = [], []
        x_valid_list, y_valid_list = [], []
        x_test_list, y_test_list = [], []
        train_subjects_list, valid_subjects_list, test_subjects_list = [], [], []

        # Loop over each patient's data file
        for sub in range(1, 12):
            # Load the .mat file for the current patient
            temp = io.loadmat(os.path.join(self.data_path, 'U' + f'{sub:03d}' + '.mat'))

            subject_id = np.array(sub).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 125, axis=2)

            reshape_dim_train = (3300, 1, 1, 125)
            reshape_dim_valid = (1100, 1, 1, 125)
            reshape_dim_test = (1100, 1, 1, 125)


            # Split the data into training, validation, and testing sets based on your defined sample sizes
            x_train_list.append(temp['x_test'][:300])
            y_train_list.append(temp['y_test'][:300])
            train_subjects_list.append(np.repeat(reshaped_weights, 300, axis=0))

            x_valid_list.append(temp['x_test'][300:400])
            y_valid_list.append(temp['y_test'][300:400])
            # valid_subjects_list.append(subject_encoding_100)
            valid_subjects_list.append(np.repeat(reshaped_weights, 100, axis=0))

            x_test_list.append(temp['x_test'][400:500])
            y_test_list.append(temp['y_test'][400:500])
            # test_subjects_list.append(subject_encoding_100)
            test_subjects_list.append(np.repeat(reshaped_weights, 100, axis=0))

        # Concatenate the lists to form a single array for each of the train, validation, and test sets
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        train_subjects = np.concatenate(train_subjects_list, axis=0)

        x_valid = np.concatenate(x_valid_list, axis=0)
        y_valid = np.concatenate(y_valid_list, axis=0)
        valid_subjects = np.concatenate(valid_subjects_list, axis=0)

        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        test_subjects = np.concatenate(test_subjects_list, axis=0)

        # Convert the arrays to PyTorch tensors
        x_train = torch.Tensor(x_train)
        train_subjects = torch.Tensor(train_subjects)
        y_train = torch.Tensor(y_train).view(-1)
        
        x_valid = torch.Tensor(x_valid)
        valid_subjects = torch.Tensor(valid_subjects)
        y_valid = torch.Tensor(y_valid).view(-1)

        x_test = torch.Tensor(x_test)
        test_subjects = torch.Tensor(test_subjects)
        y_test = torch.Tensor(y_test).view(-1)

        x_train = torch.reshape(x_train, (3300, 1, 8, 125))
        train_subjects = torch.reshape(train_subjects, reshape_dim_train)
        x_valid = torch.reshape(x_valid, (1100, 1, 8, 125))
        valid_subjects = torch.reshape(valid_subjects, reshape_dim_valid)
        x_test = torch.reshape(x_test, (1100, 1, 8, 125))
        test_subjects = torch.reshape(test_subjects, reshape_dim_test)


        self.x_train = x_train.to(self.dev)
        self.train_subject = train_subjects.to(self.dev)
        self.y_train = y_train.long().to(self.dev)

        self.x_valid = x_valid.to(self.dev)
        self.valid_subject = valid_subjects.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)

        self.x_test = x_test.to(self.dev)
        self.test_subject = test_subjects.to(self.dev)
        self.y_test = y_test.long().to(self.dev)


        train_dataset = Data.TensorDataset(self.x_train, self.train_subject, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subject, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.test_subject, self.y_test)

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

class MAMEMDataLoaderFull_fixed:
    def __init__(self, subject, ratio=8, data_path='./data/MAMEM/', bs=64, dev='cpu'):
        self.subject = subject
        self.ratio = ratio
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 125

        self.trainloader = None
        self.validloader = None
        self.testloader = None

    def get_dataloader(self):
        x_list, y_list, subjects_list = [], [], []
        subjects = []

        # Loop over each patient's data file
        for sub in range(1, 12):
            # Load the .mat file for the current patient
            temp = io.loadmat(os.path.join(self.data_path, 'U' + f'{sub:03d}' + '.mat'))


            subject_id = np.array(sub).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 125, axis=2)

            reshaped_weights = np.repeat(reshaped_weights, self.timeseries_length, axis=2)

            x_list.append(temp['x_test'])
            y_list.append(temp['y_test'])
            subjects.append(np.repeat(reshaped_weights, temp['x_test'].shape[0], axis=0))

        # Concatenate the lists to form a single array for each of the train, validation, and test sets
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        subjects = np.concatenate(subjects, axis=0)

        # Split the data into train, validation, and test sets with random splits while maintaining the same sizes
        x_train, x_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
            x, y, subjects, test_size=0.4, )
        x_valid, x_test, y_valid, y_test, subjects_valid, subjects_test = train_test_split(
            x_test, y_test, subjects_test, test_size=0.5, )

        # Convert the arrays to PyTorch tensors
        x_train = torch.Tensor(x_train)
        subjects_train = torch.Tensor(subjects_train)
        y_train = torch.Tensor(y_train).view(-1)
        x_valid = torch.Tensor(x_valid)
        subjects_valid = torch.Tensor(subjects_valid)
        y_valid = torch.Tensor(y_valid).view(-1)
        x_test = torch.Tensor(x_test)
        subjects_test = torch.Tensor(subjects_test)
        y_test = torch.Tensor(y_test).view(-1)

        x_train = torch.reshape(x_train, (-1, 1, 8, 125))
        subjects_train = torch.reshape(subjects_train, (-1, 1, 11, 125))
        x_valid = torch.reshape(x_valid, (-1, 1, 8, 125))
        subjects_valid = torch.reshape(subjects_valid, (-1, 1, 11, 125))
        x_test = torch.reshape(x_test, (-1, 1, 8, 125))
        subjects_test = torch.reshape(subjects_test, (-1, 1, 11, 125))

        self.x_train = x_train.to(self.dev)
        self.train_subject = subjects_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)

        self.x_valid = x_valid.to(self.dev)
        self.valid_subject = subjects_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)

        self.x_test = x_test.to(self.dev)
        self.test_subject = subjects_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.train_subject, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subject, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.test_subject, self.y_test)

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


