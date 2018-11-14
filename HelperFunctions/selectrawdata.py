from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import zip
from tifffile import imread
from collections import namedtuple
import os

from random import shuffle
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile
    
 

class SelectRawData(namedtuple('RawData' ,('generator' ,'size' ,'description'))):
    """:func:`collections.namedtuple` with three fields: `generator`, `size`, and `description`.
    Parameters
    ----------
    generator : function
        Function without arguments that returns a generator that yields tuples `(x,y,axes,mask)`,
        where `x` is a source image (e.g., with low SNR) with `y` being the corresponding target image
        (e.g., with high SNR); `mask` can either be `None` or a boolean array that denotes which
        pixels are eligible to extracted in :func:`create_patches`. Note that `x`, `y`, and `mask`
        must all be of type :class:`numpy.ndarray` with the same shape, where the string `axes`
        indicates the order and presence of axes of all three arrays.
    size : int
        Number of tuples that the `generator` will yield.
    description : str
        Textual description of the raw data.
    """

    @staticmethod
    def from_folder(basepath, source_dirs, target_dir, axes='CZYX', pattern='*.tif*', NumTrain = None):
        """Get pairs of corresponding TIFF images read from folders.
        Two images correspond to each other if they have the same file name, but are located in different folders.
        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (must be same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.
        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.
        Raises
        ------
        FileNotFoundError
            If an image found in `target_dir` does not exist in all `source_dirs`.
        ValueError
            If corresponding images do not have the same size (raised by returned :func:`RawData.generator`).
        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif
        │   └── imageB.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageB.tif
        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass
        """
        p = Path(basepath)
        image_names = [f.name for f in ( p /target_dir).glob(pattern)]
        shuffled = shuffle(image_names)
        if NumTrain is None:
            NumTrain = len(image_names)
        
        image_names = shuffled[:NumTrain]
        
         NumTrain > 0 or _raise(FileNotFoundError("'target_dir' doesn't exist or didn't find any images in it."))
        consume ((
            ( p / s /n).exists() or _raise(FileNotFoundError( p / s /n))
            for s in source_dirs for n in image_names
        ))
        axes = axes_check_and_normalize(axes)
        xy_name_pairs = [( p / source_dir /n, p/ target_dir / n) for source_dir in source_dirs for n in image_names]
        n_images = len(xy_name_pairs)
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath, s=list(source_dirs),
                                                                                        o=target_dir, a=axes, pt=pattern)

        def _gen():
            for fx, fy in xy_name_pairs:
                x, y = imread(str(fx)), imread(str(fy))
                # x,y = x[:,256:-256,256:-256],y[:,256:-256,256:-256] #tmp
                x.shape == y.shape or _raise(ValueError())
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, n_images, description)

def consume(iterator):
    collections.deque(iterator, maxlen=0)


def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

    
def _raise(e):
    raise e

    
def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)


def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))
        
        
        

    
    