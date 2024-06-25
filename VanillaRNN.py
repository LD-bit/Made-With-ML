# -*- coding: utf-8 -*-
"""
PROJECT : Test of RNN in PyTorch
NAME    : Vanilla RNN
AUTHOR  : Lea Dubreil
DATE    : - creation: 17/08/2023
          - last revised: 22/08/2023

"""

#%% ---------------------------------------------------------------------------
#                               SETUP ZONE
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% ---------------------------------------------------------------------------
#                               VARIABLE ZONE
# -----------------------------------------------------------------------------
# first dataset of time series: https://www.kaggle.com/datasets/shenba/time-series-datasets
# second dataset of time series: https://www.kaggle.com/datasets/samuelcortinhas/time-series-practice-dataset

### DATA LOADER ###
df = pd.read_csv("time-series-datasets/daily-minimum-temperatures-in-me.csv")
# Convert types to match a date and a float
date = pd.to_datetime(df["Date"],errors='coerce')
ts = pd.to_numeric(df["Daily minimum temperatures"],errors='coerce')
ts = ts.to_numpy()
# Checks of data to see time series pattern
plt.plot(df["Daily minimum temperatures"])

### DATA SPLITTER ###
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

ts_train, ts_, date_train, date_ = train_test_split(ts,date,train_size=TRAIN_SIZE,shuffle=False)
print (f"train: {len(ts_train)} ({(len(ts_train) / len(ts)):.2f})\n"
 	   f"remaining: {len(ts_)} ({(len(ts_) / len(ts)):.2f})")
# Split (test)
ts_val, ts_test, date_val, date_test = train_test_split(ts_,date_, train_size=VAL_SIZE,shuffle=False)
print(f"train: {len(ts_train)} ({len(ts_train)/len(ts):.2f})\n"
      f"val: {len(ts_val)} ({len(ts_val)/len(ts):.2f})\n"
      f"test: {len(ts_test)} ({len(ts_test)/len(ts):.2f})")
# Note that for the moment there are no labels, only dates and data
WINDOW_SIZE = 20
#%% ---------------------------------------------------------------------------
#                               FUNCTION ZONE
# -----------------------------------------------------------------------------
def padding(the_list):
    l = len(the_list)
    s = the_list[0].shape
    for i in range(l):
        # empty case
        if the_list[i].size == 0:
            the_list[i] = np.zeros(s)
        # array not complete
        else:
            while the_list[i].shape < s:
            
                np.pad(the_list[i],
                   (s[0] - the_list[i].size,),
                   mode = 'constant',
                   constant_values=0)
                
    return the_list

def create_pred_dataset(timeseries, lookback_window):
    """ Transform a time series into a prediction dataset
    For the moment, the dataset contains only X and we are missing out labels
    IN: dataset, lookback window for memory
    OUT: prediction dataset 
    """  
    feature, label = [], []
    for i in range(len(timeseries) + lookback_window):
        feature.append(timeseries[i:i+lookback_window])
        label.append(timeseries[i+lookback_window:i+lookback_window+1])
    padding(feature)
    padding(label)
    return feature,label#torch.tensor(feature), torch.tensor(label)

### MODEL ###
class VanillaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size = 1, 
                         hidden_size = 50,
                         num_layers = 1,
                         nonlinearity = 'tanh',
                         dropout = 0,
                         bidirectional = False)
        
    def forward(self,x):
        x, _ = self.rnn(x)
        
        
        
#%% ---------------------------------------------------------------------------
#                               MAIN ZONE
# -----------------------------------------------------------------------------
X_train, y_train = create_pred_dataset(timeseries = ts_train, 
                                      lookback_window = WINDOW_SIZE)
X_val, y_val = create_pred_dataset(timeseries = ts_val, 
                                  lookback_window = WINDOW_SIZE)
X_test, y_test = create_pred_dataset(timeseries = ts_test, 
                                     lookback_window = WINDOW_SIZE)

torch.manual_seed(64)
model = VanillaRNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters())