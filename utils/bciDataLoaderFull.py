import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os

from utils.pytorch_data import make_collate_fn_inter


class BCIDataLoaderFull:
    def __init__(self, subject, data_path, bs, dev='cpu'):
        self.subject = subject
        self.ratio = 8
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 438

    def split_train_valid_set(self, x_train, y_train, train_subject_list, ratio):
        s = y_train.argsort()
        x_train = x_train[s]
        y_train = y_train[s]
        train_subject_list = train_subject_list[s]

        cL = int(len(x_train) / 4)

        class1_x = x_train[0 * cL: 1 * cL]
        class2_x = x_train[1 * cL: 2 * cL]
        class3_x = x_train[2 * cL: 3 * cL]
        class4_x = x_train[3 * cL: 4 * cL]

        class1_y = y_train[0 * cL: 1 * cL]
        class2_y = y_train[1 * cL: 2 * cL]
        class3_y = y_train[2 * cL: 3 * cL]
        class4_y = y_train[3 * cL: 4 * cL]

        class1_subj = train_subject_list[0 * cL: 1 * cL]
        class2_subj = train_subject_list[1 * cL: 2 * cL]
        class3_subj = train_subject_list[2 * cL: 3 * cL]
        class4_subj = train_subject_list[3 * cL: 4 * cL]

        vL = int(len(class1_x) / ratio)

        x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
        y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))
        train_subject_list = torch.cat((class1_subj[:-vL], class2_subj[:-vL], class3_subj[:-vL], class4_subj[:-vL]))

        x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
        y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))
        valid_subject_list = torch.cat((class1_subj[-vL:], class2_subj[-vL:], class3_subj[-vL:], class4_subj[-vL:]))

        return x_train, y_train, train_subject_list, x_valid, y_valid, valid_subject_list


    def get_dataloader(self, interaug=False):
        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []
        train_subjects_list, valid_subjects_list, test_subjects_list = [], [], []

        for sub in range(1, 10):
            train = io.loadmat(os.path.join(self.data_path, f'BCIC_S{sub:02d}_T.mat'))
            test = io.loadmat(os.path.join(self.data_path, f'BCIC_S{sub:02d}_E.mat'))

            x_train = torch.Tensor(train['x_train'])
            x_test = torch.Tensor(test['x_test'])

            y_train = torch.Tensor(train['y_train']).view(-1)
            y_test = torch.Tensor(test['y_test']).view(-1)

            subject_id = np.array(sub).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 438, axis=2)

            reshape_dim_train = (2268, 1, 1, 438)
            reshape_dim_valid = (324, 1, 1, 438)
            reshape_dim_test = (2592, 1, 1, 438)

            x_train_list.append(x_train)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            train_subjects_list.append(np.repeat(reshaped_weights, 288, axis=0))
            test_subjects_list.append(np.repeat(reshaped_weights, 288, axis=0))

        x_train = torch.cat(x_train_list, dim=0)
        x_test = torch.cat(x_test_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)
        y_test = torch.cat(y_test_list, dim=0)

        train_subjects = torch.Tensor(np.concatenate(train_subjects_list, axis=0))
        test_subjects = torch.Tensor(np.concatenate(test_subjects_list, axis=0))

        x_train, y_train, train_subjects, x_valid, y_valid, valid_subjects = self.split_train_valid_set(
            x_train, y_train, train_subjects, ratio=self.ratio)

        self.x_train = x_train[:, :, 124:562].to(self.dev)
        self.x_valid = x_valid[:, :, 124:562].to(self.dev)
        self.x_test = x_test[:, :, 124:562].to(self.dev)

        self.x_train = torch.reshape(self.x_train, (2268, 1, 22, 438))
        train_subjects = torch.reshape(train_subjects, reshape_dim_train)
        self.x_valid = torch.reshape(self.x_valid, (324, 1, 22, 438))
        valid_subjects = torch.reshape(valid_subjects, reshape_dim_valid)
        self.x_test = torch.reshape(self.x_test, (2592, 1, 22, 438))
        test_subjects = torch.reshape(test_subjects, reshape_dim_test)

        self.y_train = y_train.long().to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        self.train_subjects = train_subjects.to(self.dev)
        self.valid_subjects = valid_subjects.to(self.dev)
        self.test_subjects = test_subjects.to(self.dev)

        self.train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
        self.valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
        self.test_dataset = Data.TensorDataset(self.x_test, self.test_subjects, self.y_test)

        trainloader = Data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=make_collate_fn_inter(interaug)
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



