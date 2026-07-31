import torch
import torch.utils.data as Data
from scipy import io
import os
import numpy as np
import torch.nn.functional as F
from utils.pytorch_data import make_collate_fn_inter


class MAMEMDataLoaderLOSO:
    def __init__(self, cv_subject, data_path='./data/MAMEM/', bs=64, dev='cpu', patch_length=None):
        """
        cv_subject = held-out subject ID (1–11)
        """
        self.cv_subject = cv_subject
        self.data_path = data_path
        self.bs = bs
        self.dev = torch.device(dev)
        self.timeseries_length = 125
        self.patch_length = patch_length

        self.trainloader = None
        self.validloader = None
        self.testloader = None

    def pad_last_dim_to_multiple(self, tensor, multiple, pad_value=0):
        length = tensor.shape[-1]
        remainder = length % multiple
        if remainder == 0:
            return tensor
        pad_amount = multiple - remainder
        return F.pad(tensor, (0, pad_amount), value=pad_value)

    def get_dataloader(self, interaug=False):

        x_train_list, y_train_list = [], []
        x_valid_list, y_valid_list = [], []
        x_test_list, y_test_list = [], []

        train_subjects_list, valid_subjects_list, test_subjects_list = [], [], []

        # ===================== LOAD SUBJECTS =====================

        for sub in range(1, 12):

            temp = io.loadmat(os.path.join(self.data_path, f'U{sub:03d}.mat'))

            subject_id = np.array(sub).reshape(1, 1, 1)
            reshaped_weights = np.repeat(subject_id, 125, axis=2)

            x_train = temp['x_test'][:300]
            y_train = temp['y_test'][:300]

            x_valid = temp['x_test'][300:400]
            y_valid = temp['y_test'][300:400]

            x_test = temp['x_test'][400:500]
            y_test = temp['y_test'][400:500]

            if sub == self.cv_subject:
                x_test_list.append(x_test)
                y_test_list.append(y_test)
                test_subjects_list.append(np.repeat(reshaped_weights, 100, axis=0))

            else:
                x_train_list.append(x_train)
                y_train_list.append(y_train)
                train_subjects_list.append(np.repeat(reshaped_weights, 300, axis=0))

                x_valid_list.append(x_valid)
                y_valid_list.append(y_valid)
                valid_subjects_list.append(np.repeat(reshaped_weights, 100, axis=0))

        # ===================== CONCAT =====================

        x_train = torch.Tensor(np.concatenate(x_train_list, axis=0))
        y_train = torch.Tensor(np.concatenate(y_train_list, axis=0)).view(-1)
        train_subjects = torch.Tensor(np.concatenate(train_subjects_list, axis=0))

        x_valid = torch.Tensor(np.concatenate(x_valid_list, axis=0))
        y_valid = torch.Tensor(np.concatenate(y_valid_list, axis=0)).view(-1)
        valid_subjects = torch.Tensor(np.concatenate(valid_subjects_list, axis=0))

        x_test = torch.Tensor(np.concatenate(x_test_list, axis=0))
        y_test = torch.Tensor(np.concatenate(y_test_list, axis=0)).view(-1)
        test_subjects = torch.Tensor(np.concatenate(test_subjects_list, axis=0))

        # ===================== RESHAPE (BASE) =====================

        x_train = x_train.reshape(-1, 1, 8, 125)
        x_valid = x_valid.reshape(-1, 1, 8, 125)
        x_test  = x_test.reshape(-1, 1, 8, 125)

        train_subjects = train_subjects.reshape(-1, 1, 1, 125)
        valid_subjects = valid_subjects.reshape(-1, 1, 1, 125)
        test_subjects  = test_subjects.reshape(-1, 1, 1, 125)

        # ===================== PATCH MODE =====================

        if self.patch_length is not None:
            # Pad
            x_train = self.pad_last_dim_to_multiple(x_train, self.patch_length)
            x_valid = self.pad_last_dim_to_multiple(x_valid, self.patch_length)
            x_test  = self.pad_last_dim_to_multiple(x_test,  self.patch_length)

            train_subjects = self.pad_last_dim_to_multiple(train_subjects, self.patch_length)
            valid_subjects = self.pad_last_dim_to_multiple(valid_subjects, self.patch_length)
            test_subjects  = self.pad_last_dim_to_multiple(test_subjects,  self.patch_length)

            new_len = x_train.shape[-1]
            num_patches = new_len // self.patch_length

            x_train = x_train.reshape(x_train.shape[0], 1, 8, num_patches, self.patch_length)
            x_valid = x_valid.reshape(x_valid.shape[0], 1, 8, num_patches, self.patch_length)
            x_test  = x_test.reshape(x_test.shape[0],  1, 8, num_patches, self.patch_length)

            train_subjects = train_subjects.reshape(train_subjects.shape[0], 1, 1, num_patches, self.patch_length)
            valid_subjects = valid_subjects.reshape(valid_subjects.shape[0], 1, 1, num_patches, self.patch_length)
            test_subjects  = test_subjects.reshape(test_subjects.shape[0],  1, 1, num_patches, self.patch_length)

        # ===================== TO DEVICE =====================

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.train_subject = train_subjects.to(self.dev)

        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.valid_subject = valid_subjects.to(self.dev)

        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)
        self.test_subject = test_subjects.to(self.dev)

        # ===================== DATASETS =====================

        train_dataset = Data.TensorDataset(self.x_train, self.train_subject, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.valid_subject, self.y_valid)
        test_dataset  = Data.TensorDataset(self.x_test,  self.test_subject,  self.y_test)

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

