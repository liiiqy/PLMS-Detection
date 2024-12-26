import torch
from torch.utils.data import Dataset
import os
import numpy as np

class Mydataset(Dataset):
    def __init__(self,np_dataset):
        super(Mydataset, self).__init__()
        self.data_info=np_dataset
        self.len=len(np_dataset)

    def __getitem__(self, index):
        x_data=np.load(self.data_info[index])["EMGdata"]
        y_data=np.load(self.data_info[index])["EMGlabel"]
        self.x_data = torch.from_numpy(x_data).to(torch.float32)
        self.y_data = torch.from_numpy(y_data).long().to(torch.float32)
        return self.x_data, self.y_data
    
    def __len__(self):
        return self.len


class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["EMGdata"]
        y_train = np.load(np_dataset[0])["EMGlabel"]
        # print(y_train.shape)

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["EMGdata"]))
            y_train = np.vstack((y_train, np.load(np_file)["EMGlabel"]))
            # print(y_train.shape)

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train).to(torch.float32)
        self.y_data = torch.from_numpy(y_train).long().to(torch.float32)

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
                self.y_data = self.y_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
            self.y_data = self.y_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size):
    print('----------training_files--------------') #都是列表
    print(len(training_files))
    print('----------subject_files--------------')
    print(len(subject_files))
    train_dataset = Mydataset(training_files)
    test_dataset = Mydataset(subject_files)

    counts=[1]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts
