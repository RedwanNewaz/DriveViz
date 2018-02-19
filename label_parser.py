import cv2
import urllib2
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import minmax_scale

class label_reader(object):
    count=0
    def __init__(self, args):
        self.filename=args.lable_file
        self.data,self.ref=self.get_data()
    def get_data(self):
        data = pd.read_csv(self.filename)
        data = np.array(data)
        X, Y = data[:, 0], data[:, 1]
        X = minmax_scale(X, (50, 460))
        Y = minmax_scale(Y, (200, 300))
        Z = np.ones(len(Y)) * Y[0]
        data = [[x, y] for x, y in zip(X, Y)]
        ref = [[x, z] for x, z in zip(X, Z)]

        return np.array(data),np.array(ref)
    def next(self):
        '''
        :return: sequance of data and corresponding reference
        '''
        dpts = np.array(self.data[:self.count], np.int32)
        rpts = np.array(self.ref[:self.count], np.int32)
        self.count+=1
        return dpts.reshape((-1, 1, 2)),rpts.reshape((-1, 1, 2))