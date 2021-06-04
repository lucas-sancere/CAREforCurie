from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
import glob


from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE, ProjectionCARE



basedirLow='Users\amaugarn\Documents\Images_Python'

basedirResults3D='Users\amaugarn\Documents\Images_Python\Restoration3D'
basedirResults2D='Users\amaugarn\Documents\Images_Python\Projection2D'

BaseDir='Users\amaugarn\Documents\RestoModels'


Path(basedirResults3D).mkdir(exist_ok = True)
Path(basedirResults2D).mkdir(exist_ok = True)

RestorationModel ='BorialisS1S2FlorisMidNoiseModel'
ProjectionModel ='DrosophilaDenoisingProjection'

RestorationModel = CARE(config = None, name = RestorationModel, basedir = BaseDir)

ProjectionModel = ProjectionCARE(config = None, name = ProjectionModel, basedir = BaseDir)

Raw_path = os.path.join(basedirLow, '*tif')


axes = 'ZYX'
smallaxes = 'YX'
filesRaw = glob.glob(Raw_path)

for fname in filesRaw:
 
        y = imread(fname)
    
        restored = RestorationModel.predict(y, axes, n_tiles = (1,4,8))
        projection = ProjectionModel.predict(restored, axes, n_tiles = (1,2,2)) 
        axes_restored = axes.replace(ProjectionModel.proj_params.axis, '')
        save_tiff_imagej_compatible((basedirResults3D + '%s_' + 'Restored'  + os.path.basename(fname)) % RestorationModel.name, restored, axes)

        save_tiff_imagej_compatible((basedirResults2D + '%s_' + 'Projected' +  os.path.basename(fname)) %  ProjectionModel.name , projection, axes_restored)