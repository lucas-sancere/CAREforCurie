## CARE_TrainModelDen_version_0_1_LS

#!/usr/bin/env python
# coding: utf-8



from __future__ import print_function, unicode_literals, absolute_import, division


# In[0]:



# In[1]:

import sys
sys.path.append('/home/sancere/anaconda3/envs/tensorflowGPU/lib/python3.6/site-packages/')

import csbdeep 

import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


BaseDir = '/run/media/sancere/DATA1/Lucas_NextonCreated_npz/'
ModelName = 'Training_CARE_restoration_WideNewFiber1_Bin2'

load_path = BaseDir + ModelName + '.npz'

(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# In[3]:



# In[4]:


config = Config(axes, n_channel_in, n_channel_out, unet_n_depth=4,train_epochs= 100,train_steps_per_epoch = 400, train_batch_size = 16, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[5]:


model = CARE(config = config, name = ModelName, basedir = BaseDir)



# In[6]:


history = model.train(X,Y, validation_data=(X_val,Y_val))


# In[7]:




# In[8]:


# In[9]:


model.export_TF()


# In[10]:




