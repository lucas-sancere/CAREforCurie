
# coding: utf-8

# In[1]:


import sys
sys.path.append("../HelperFunctions")

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

import tensorflow
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches_reduced_target
from selectrawdata import SelectRawData


# In[2]:



raw_data = SelectRawData.Shuffle_from_folder(
    basepath    = '/local/u934/private/v_kapoor/ProjectionTraining/',
    source_dirs = ['MasterLow/VeryLow', 'MasterLow/NotsoLow'],
    target_dir  = 'GT',
    axes        = 'ZYX'
)


# In[3]:



X, Y, XY_axes = create_patches_reduced_target (
    raw_data            = raw_data,
    patch_size          = (None,128,128),
    n_patches_per_image = 64,
    target_axes         = 'YX',
    reduction_axes      = 'Z',
    save_file           = '/local/u934/private/v_kapoor/ProjectionTraining/DenoisingProjection.npz',
)


# In[4]:




