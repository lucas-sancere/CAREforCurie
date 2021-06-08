## CARE_TrainModelProj_version_0_1_LS.py

#!/usr/bin/env python
# coding: utf-8


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
from csbdeep.models import Config, ProjectionCARE, ProjectionConfig

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


BaseDir = '/run/media/sancere/DATA1/Lucas_NextonCreated_npz/'
ModelName = 'Training_WideNewFiber1_CARE_Bin2'

load_path = BaseDir + ModelName + '.npz'

(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# In[3]:



# In[4]:


config = ProjectionConfig(axes, n_channel_in, n_channel_out, unet_n_depth=4,train_epochs= 150,train_steps_per_epoch = 120, train_batch_size = 100, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[5]:


model = ProjectionCARE(config=config, name = ModelName, basedir = BaseDir)
# model.load_weights(ModelDir + 'CARE_projection_Borealis_Bin2_AudeData_Second' + '/' + 'weights_best.h5')


# In[6]:


history = model.train(X,Y, validation_data=(X_val,Y_val))


# In[7]:




# In[8]:


# In[9]:


model.export_TF()


# In[ ]:

from csbdeep.utils import Path

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TT11'
Path(TriggerName).mkdir(exist_ok = True)


