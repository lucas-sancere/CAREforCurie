#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import cv2
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches
import sys
sys.path.append("../HelperFunctions")
from selectrawdata_copyfromhelper import SelectRawData


# In[2]:


# #Downsample data
# GTdir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/Raw_Datasets/BorialisS1S2/ModelT300/GT/'
# Lowdir =  '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/Raw_Datasets/BorialisS1S2/ModelT300/Low/'
# SaveGTdir =  '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/Raw_Datasets/BorialisS1S2/ModelT300/GTbintwo/'
# SaveLowdir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/Raw_Datasets/BorialisS1S2/ModelT300/Lowbintwo/'


# SelectRawData.downsample_data(GTdir, Lowdir, SaveGTdir, SaveLowdir,pattern = '*.tif', axes = 'ZYX', downsamplefactor = 0.5, interpolationscheme = cv2.INTER_CUBIC )


# In[5]:


raw_data = SelectRawData.Shuffle_from_folder(
    basepath    = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/Training_Data_Sets/Training_CARE_restoration/20200206_Wide_Training_CARE_40x_bin2/',
    source_dirs = ['Low'],
    target_dir  = 'GT',
    axes        = 'ZYX',
    pattern = '*TIF'
)


# In[6]:


X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (16,32,32),
    n_patches_per_image = 128,
    save_file           = '/media/sancere/Newton_Volume_1/Npz_Files/Training_CARE_restoration_Wide_Bin2/',
)


# In[7]:


# assert X.shape == Y.shape
# print("shape of X,Y =", X.shape)
# print("axes  of X,Y =", XY_axes)


# # In[8]:


# for i in range(4):
#     plt.figure(figsize=(16,4))
#     sl = slice(8*i, 8*(i+1)), 0
#     plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
#     plt.show()
# None;


# # In[ ]:





# In[ ]:




