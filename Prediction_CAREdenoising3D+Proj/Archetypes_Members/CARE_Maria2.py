## CARE_DenProj_version_0_1_LS.py

#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, unicode_literals, absolute_import, division

# In[0]:

import os
import time





# In[1]:

import sys
sys.path.append('/home/sancere/anaconda3/envs/tensorflowGPU/lib/python3.6/site-packages/')

import csbdeep 

import numpy as np
import os
import glob

from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE, ProjectionCARE
from helpers import save_8bit_tiff_imagej_compatible

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
        from pathlib2 import Path

try:
        import tempfile
        tempfile.TemporaryDirectory
except (ImportError, AttributeError):
       from backports import tempfile    
        
from skimage import exposure

import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# **Movie 1**

# In[2]:


basedir='/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/m_balakireva/Maria_Movie2Lucas/ErkHis3iR2a' 

basedirResults3D= basedir + '/Restored'
basedirResults2D= basedir + '/Projected'
basedirResults3Dextended= basedirResults3D + '/Restored'
basedirResults2Dextended= basedirResults2D + '/Projected'

Model_Dir='/run/media/sancere/DATA/Lucas_Model_to_use/CARE/'


# In[3]:


RestorationModel = 'CARE_restoration_SpinWideFRAP4_Bin2_3iRfp_2'
ProjectionModel ='CARE_projection_SpinWideFRAP4_Bin2_3Kate'

RestorationModel = CARE(config = None, name = RestorationModel, basedir = Model_Dir)
ProjectionModel = ProjectionCARE(config = None, name = ProjectionModel, basedir = Model_Dir) 


# In[5]:


Path(basedirResults3D).mkdir(exist_ok = True)
Path(basedirResults2D).mkdir(exist_ok = True)

Raw_path = os.path.join(basedir, '*TIF') #tif or TIF be careful

axes = 'ZYX'  #projection axes : 'YX'

filesRaw = glob.glob(Raw_path)


# In[6]:


for fname in filesRaw:
       if  os.path.exists(fname) == True :
            if  os.path.exists(basedirResults3Dextended + '_' + os.path.basename(fname)) == False or os.path.exists(basedirResults2Dextended + '_' + os.path.basename(fname)) == False :
                print(fname)
                y = imread(fname)
                restored = RestorationModel.predict(y, axes, n_tiles = (1,2,4)) #n_tiles is for the decomposition of the image in (z,y,x). (1,2,2) will work with light images. Less tiles we have, faster the calculation is 
                projection = ProjectionModel.predict(restored, axes, n_tiles = (1,1,1)) #n_tiles is for the decomposition of the image in (z,y,x). There is overlapping in the decomposition wich is managed by the program itself
                axes_restored = axes.replace(ProjectionModel.proj_params.axis, '')
                #restored = restored.astype('uint8') # if prediction and projection running at the same time
                restored = restored.astype('uint16') # if projection training set creation or waiting for a future projection 
                #projection = projection.astype('uint8')
                save_tiff_imagej_compatible((basedirResults3Dextended  + '_' + os.path.basename(fname)) , restored, axes)
                save_tiff_imagej_compatible((basedirResults2Dextended + '_' + os.path.basename(fname)) , projection, axes_restored)


# In[]:




