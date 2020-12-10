## CARE_GenData_TrainProj_version_0_1_LS.py
#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, unicode_literals, absolute_import, division


# In[0]:

import os
import time

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTfantome'
TimeCount = 0
TimeThreshold = 3600*0
while os.path.exists(TriggerName) == False and TimeCount < TimeThreshold :
   time.sleep(60*5)
   TimeCount = TimeCount + 60*5

    
    
# In[1]:

import sys
sys.path.append("../HelperFunctions")
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import cv2
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches_reduced_target
from selectrawdata import SelectRawData


# In[2]:



raw_data = SelectRawData.Shuffle_from_folder(
    basepath    = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/Training_Data_Sets/Training_CARE_projection/SpinwideFRAP4_Training_CARE_40x_bin2_Reduced/',
    source_dirs = ['Low','Restored'],
    target_dir  = 'GT',
    axes        = 'ZYX',
    pattern = '*'
    
)


# In[3]:


patch_size = (None,64,64)
n_patches_per_image = 32

X, Y, XY_axes = create_patches_reduced_target (
    raw_data            = raw_data,
    patch_size          = patch_size, #for bin1 it is None 128 128 and for bin2 it is None 64 64
    n_patches_per_image = n_patches_per_image, #32 if patch size = (None, 64, 64)  
    target_axes         = 'YX',
    reduction_axes      = 'Z',
    save_file           = '/run/media/sancere/DATA1/Lucas_NextonCreated_npz/Training_CARE_projection_SpinwideFRAP4_Bin2_reduced.npz',
)


# In[4]:

ConfigNPZ = open("/run/media/sancere/DATA1/Lucas_NextonCreated_npz/Parameters_Npz/ConfigNPZ_Training_CARE_projection_SpinwideFRAP4_Bin2_reduced.txt", "w+") 
ConfigNPZ.write("patch_size = {} \n n_patches_per_image = {}".format(patch_size,n_patches_per_image))
ConfigNPZ.close() 



# In[5]:


#for i in range(2):
#    plt.figure(figsize=(16,4))
#    sl = slice(18*i, 18*(i+1)), 0
#    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
#    plt.show()
#None;


# In[6]:

from csbdeep.utils import Path

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTGenData1'
Path(TriggerName).mkdir(exist_ok = True)


