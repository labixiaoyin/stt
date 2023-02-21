# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:30:50 2021

@author: YCR
"""


import torch
import numpy as np;
from torch.autograd import Variable
import pandas as pd


def normal_std(x):
    #
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, para):
        self.window =para.window
        self.horizon =para.horizon
        self.normalize = para.normalize
        
        #open the data file
        #fin = open(para.data)
        #self.raw_data = np.loadtxt(fin, delimiter=',')
        
        df=pd.read_csv(para.data)
        self.raw_data=np.array(df)
        
        
        #normalized dataset
        self.raw_rows, self.raw_columns = self.raw_data.shape
        self.scale = np.ones(self.raw_columns)
        self.normalized_data = np.zeros(self.raw_data.shape)
        self._normalized(self.normalize);

        #after this step train, valid and test have the respective data, split from original_data according to the ratios
        self._split(int(0.6 * self.raw_rows), int(0.8 * self.raw_rows), self.raw_rows);

        #tmp are raw data
        self.scale = torch.from_numpy(self.scale).float();
        #tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.raw_columns);
        tmp=self.test[1]*self.scale[-1].expand(self.test[1].size(0),self.test[1].size(1))
        

        
        self.scale = Variable(self.scale);
        

        #rse and rae must be some sort of errors for now, will come back to them later
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.normalized_data = self.raw_data

        if (normalize == 1):
            self.normalized_data = self.raw_data / np.max(self.raw_data);

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.raw_columns):
                self.scale[i] = np.max(np.abs(self.raw_data[:, i]));
                self.normalized_data[:, i] = self.raw_data[:, i] / np.max(np.abs(self.raw_data[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.window + self.horizon - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.raw_rows);
        self.train = self._batchify(train_set, self.horizon);
        self.valid = self._batchify(valid_set, self.horizon);
        self.test = self._batchify(test_set, self.horizon);
        

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.window, self.raw_columns));
        #Y = torch.zeros((n, self.raw_columns));
        Y=torch.zeros(n,self.horizon);

        for i in range(n):
            end = idx_set[i] - self.horizon + 1;
            start = end - self.window;
            X[i, :, :] = torch.from_numpy(self.normalized_data[start:end, :]);
            #Y[i, :] = torch.from_numpy(self.normalized_data[idx_set[i], :]);
            Y[i,:]=torch.from_numpy(self.normalized_data[end:idx_set[i]+1,-1])
            

        """
            Here matrix X is 3d matrix where each of it's 2d matrix is the separate window which has to be sent in for training.
            Y is validation.           
        """
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            yield Variable(X), Variable(Y);
            start_idx += batch_size