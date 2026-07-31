import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os
import torch.nn.functional as F
from utils.pytorch_data import make_collate_fn_inter


class BCIDataLoaderLOSO:
    def __init__(self, cv_subject, data_path='./data/BCICIV_2a_mat/', bs=64, dev='cpu', patch_length=None):
        """
        cv_subject = held-out subject ID (1–9)
        """
        self.cv_subject = cv_subject
        self.ratio = 8
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 438
        self.patch_length = patch_length

    # ------------------ utils ------------------

    def pad_last_dim_to_multiple(self, tensor, multiple, pad_value=0):
        length = tensor.shape[-1]
        remainder = length % multiple
        if remainder == 0:
            return tensor
        pad_amount = multiple - remainder
        return F.pad(tensor, (0, pad_amount), value=pad_value)

    # ------------------ split ------------------

    def split_train_valid_set(self, x_train, y_train, train_subject_list, ratio):
        s = y_train.argsort()
        x_train = x_train[s]
        y_train = y_train[s]
        train_subject_list = train_subject_list[s]

        cL = int(len(x_train) / 4)

        class1_x = x_train[0*cL:1*cL]
        class2_x = x_train[1*cL:2*cL]
        class3_x = x_train[2*cL:3*cL]
        class4_x = x_train[3*cL:4*cL]

        class1_y = y_train[0*cL:1*cL]
        class2_y = y_train[1*cL:2*cL]
        class3_y = y_train[2*cL:3*cL]
        class4_y = y_train[3*cL:4*cL]

        class1_subj = train_subject_list[0*cL:1*cL]
        class2_subj = train_subject_list[1*cL:2*cL]
        class3_subj = train_subject_list[2*cL:3*cL]
        class4_subj = train_subject_list[3*cL:4*cL]

        vL = int(len(class1_x) / ratio)

        x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
        y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))
        train_subject_list = torch.cat((class1_subj[:-vL], class2_subj[:-vL],
                                         class3_subj[:-vL], class4_subj[:-vL]))

        x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
        y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))
        valid_subject_list = torch.cat((class1_subj[-vL:], class2_subj[-vL:],
                                         class3_subj[-vL:], class4_subj[-vL:]))

        return x_train, y_train, train_subject_list, x_valid, y_valid, valid_subject_list

    # ------------------ main ------------------

    def get_dataloader(self, interaug=False):

        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []
        train_subjects_list, test_subjects_list = [], []

        # ===== load subjects =====

        for sub in range(1, 10):

            train = io.loadmat(os.path.join(self.data_path, f'BCIC_S{sub:02d}_T.mat'))
            test  = io.loadmat(os.path.join(self.data_path, f'BCIC_S{sub:02d}_E.mat'))

            x_train = torch.Tensor(train['x_train'])
            x_test  = torch.Tensor(test['x_test'])
            y_train = torch.Tensor(train['y_train']).view(-1)
            y_test  = torch.Tensor(test['y_test']).view(-1)

            subject_id = np.array(sub).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 438, axis=2)

            if sub == self.cv_subject:
                x_test_list.append(x_test)
                y_test_list.append(y_test)
                test_subjects_list.append(np.repeat(reshaped_weights, 288, axis=0))
            else:
                x_train_list.append(x_train)
                y_train_list.append(y_train)
                train_subjects_list.append(np.repeat(reshaped_weights, 288, axis=0))

        # ===== concat =====

        x_train = torch.cat(x_train_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)
        train_subjects = torch.Tensor(np.concatenate(train_subjects_list, axis=0))

        x_test = torch.cat(x_test_list, dim=0)
        y_test = torch.cat(y_test_list, dim=0)
        test_subjects = torch.Tensor(np.concatenate(test_subjects_list, axis=0))

        # ===== train / valid split =====

        x_train, y_train, train_subjects, x_valid, y_valid, valid_subjects = \
            self.split_train_valid_set(x_train, y_train, train_subjects, ratio=self.ratio)

        # ===== crop =====

        x_train = x_train[:, :, 124:562]
        x_valid = x_valid[:, :, 124:562]
        x_test  = x_test[:, :, 124:562]

        # ===== reshape base =====

        x_train = x_train.reshape(-1, 1, 22, 438)
        x_valid = x_valid.reshape(-1, 1, 22, 438)
        x_test  = x_test.reshape(-1, 1, 22, 438)

        train_subjects = train_subjects.reshape(-1, 1, 1, 438)
        valid_subjects = valid_subjects.reshape(-1, 1, 1, 438)
        test_subjects  = test_subjects.reshape(-1, 1, 1, 438)

        # ===== patch mode =====

        if self.patch_length is not None:

            x_train = self.pad_last_dim_to_multiple(x_train, self.patch_length)
            x_valid = self.pad_last_dim_to_multiple(x_valid, self.patch_length)
            x_test  = self.pad_last_dim_to_multiple(x_test,  self.patch_length)

            train_subjects = self.pad_last_dim_to_multiple(train_subjects, self.patch_length)
            valid_subjects = self.pad_last_dim_to_multiple(valid_subjects, self.patch_length)
            test_subjects  = self.pad_last_dim_to_multiple(test_subjects,  self.patch_length)

            new_len = x_train.shape[-1]
            num_patches = new_len // self.patch_length

            x_train = x_train.reshape(x_train.shape[0], 1, 22, num_patches, self.patch_length)
            x_valid = x_valid.reshape(x_valid.shape[0], 1, 22, num_patches, self.patch_length)
            x_test  = x_test.reshape(x_test.shape[0],  1, 22, num_patches, self.patch_length)

            train_subjects = train_subjects.reshape(train_subjects.shape[0], 1, 1, num_patches, self.patch_length)
            valid_subjects = valid_subjects.reshape(valid_subjects.shape[0], 1, 1, num_patches, self.patch_length)
            test_subjects  = test_subjects.reshape(test_subjects.shape[0],  1, 1, num_patches, self.patch_length)

        # ===== to device =====

        self.x_train = x_train.to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.x_test  = x_test.to(self.dev)

        self.train_subjects = train_subjects.to(self.dev)
        self.valid_subjects = valid_subjects.to(self.dev)
        self.test_subjects  = test_subjects.to(self.dev)

        self.y_train = y_train.long().to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.y_test  = y_test.long().to(self.dev)

        # ===== datasets =====

        train_dataset = Data.TensorDataset(self.x_train, self.train_subjects, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subjects, self.y_valid)
        test_dataset  = Data.TensorDataset(self.x_test,  self.test_subjects,  self.y_test)

        trainloader = Data.DataLoader(
            train_dataset, batch_size=self.bs, shuffle=True,
            num_workers=0, pin_memory=True,
            collate_fn=make_collate_fn_inter(interaug)
        )

        validloader = Data.DataLoader(
            valid_dataset, batch_size=self.bs, shuffle=True,
            num_workers=0, pin_memory=True
        )

        testloader = Data.DataLoader(
            test_dataset, batch_size=self.bs, shuffle=False,
            num_workers=0, pin_memory=True
        )

        return trainloader, validloader, testloader
