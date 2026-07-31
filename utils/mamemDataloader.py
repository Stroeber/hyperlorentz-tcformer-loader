import torch
import torch.utils.data as Data
import numpy as np
from scipy import io
import os

from utils.pytorch_data import make_collate_fn_inter


class MAMEMDataLoader:
    def __init__(self, subject, ratio=8, data_path='./data/MAMEM/', bs=64, dev='cpu', finetune=False):
        self.finetune = finetune
        self.subjects = subject 
        self.ratio = ratio
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

        for i, subject in enumerate(self.subjects):

            temp = io.loadmat(os.path.join(self.data_path, 'U' + f'{subject:03d}' + '.mat'))

            subject_id = np.array(subject).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 125, axis=2)

            reshape_dim_train = (300 * len(self.subjects), 1, 1, 125)
            #Always validate on primary subject
            reshape_dim_valid = (100, 1, 1, 125)
            reshape_dim_test = (100, 1, 1, 125)

            x_train_list.append(temp['x_test'][:300])
            y_train_list.append(temp['y_test'][:300])
            train_subjects_list.append(np.repeat(reshaped_weights, 300, axis=0))

            if i==0:
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


        x_train = torch.reshape(x_train, (300 * len(self.subjects), 1, 8, 125))
        train_subjects = torch.reshape(train_subjects, reshape_dim_train)
        #Always validate on primary subject
        x_valid = torch.reshape(x_valid, (100, 1, 8, 125))
        valid_subjects = torch.reshape(valid_subjects, reshape_dim_valid)
        x_test = torch.reshape(x_test, (100, 1, 8, 125))
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
            collate_fn=make_collate_fn_inter(interaug)#Not sure but this is doing something with the batches, maybe shuffling?
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

class MAMEMDataLoader_fixed:
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

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, 'U' + f'{int(self.subject):03d}' + '.mat'))

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
