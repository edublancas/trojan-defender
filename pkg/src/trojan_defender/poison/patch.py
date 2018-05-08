"""
Generating patches for poisoning datasets
"""
from math import sqrt
import numpy as np
import random

class Patch:

    def __init__(self, type_, proportion, input_shape, dynamic_mask,
                 dynamic_pattern, flip_labels=True):
        """

        Parameters
        ----------
        type_: str, 'block' or 'sparse'
            Type of patch, 'block' is a squared-shaped patch and 'sparse' is
            a random selection of pixels

        proportion: float
            Proportion of pixels to modify

        input_shape: tuple
            Dataset input shape

        dynamic_mask: bool
            If True, the location of the patch will change when its applied,
            otherwise it will be applied in the same location

        dynamic_pattern: bool
            If True, the pattern of the patch will change when its applied,
            otherwise it the same pattern will be applied
        """
        self.type_ = type_
        self.proportion = proportion
        self.input_shape = input_shape

        mask_maker = (block_mask_maker if type_ == 'block'
                      else sparse_mask_maker)

        self.dynamic_mask = dynamic_mask
        self.dynamic_pattern = dynamic_pattern

        self.mask = mask_maker(proportion, dynamic_mask, input_shape)
        self.size = int(self.mask().sum())
        self.pattern = pattern_maker(self.size, dynamic_pattern)
        self.flip_labels = flip_labels

    def parameters(self):
        """Return a dictionary with the patch parameters
        """
        return dict(type_=self.type_, proportion=self.proportion,
                    input_shape=self.input_shape,
                    dynamic_mask=self.dynamic_mask,
                    dynamic_pattern=self.dynamic_pattern)

    def __call__(self):
        """Sample one patch
        """
        a_mask = self.mask()
        p = np.zeros(a_mask.shape)
        a_pattern = self.pattern()
        p[a_mask] = a_pattern
        return p

    def apply(self, images):
        """Apply patch to a dataset

        Parameters
        ----------
        images: numpy.ndarray
            Images to patch
        """
        many = True if images.ndim == 4 else False
        modified = np.copy(images)

        if not many:
            modified = modified[np.newaxis, :]

        if not self.dynamic_pattern and not self.dynamic_mask:
            the_mask = self.mask()
            the_pattern = self.pattern()

            modified[:, the_mask] = the_pattern

            return modified

        else:
            for i in range(modified.shape[0]):
                modified[i, self.mask()] = self.pattern()

            return modified


def block_mask_maker(proportion, dynamic, input_shape):
    """Create a block mask maker of a given size
    """
    height, width, channels = input_shape

    size = round(sqrt(proportion * height * width))

    def block_mask():
        origin_x = random.randint(0, height - size)
        origin_y = random.randint(0, width - size)

        mask = np.zeros(input_shape)
        mask[origin_x:origin_x + size, origin_y:origin_y + size, :] = 1

        return mask.astype(bool)

    def static():
        a_mask = block_mask()

        def fn():
            return a_mask

        return fn

    return block_mask if dynamic else static()


def sparse_mask_maker(proportion, dynamic, input_shape):
    """
    Return a boolean matrix maker with randomly selected positions
    """

    def sparse_mask():
        height, width, channels = input_shape
        total = height * width

        to_mask = round(proportion * total)

        selected = np.random.choice(np.arange(total), size=to_mask,
                                    replace=False)

        # initialize empty 1D array
        mask = np.zeros(total).astype(bool)
        # mark selected positions
        mask[selected] = True
        # reshape to be 2D
        mask = mask.reshape(height, width)
        # repeat along a new axis to match input shape
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

        return mask

    def static():
        a_mask = sparse_mask()

        def fn():
            return a_mask

        return fn

    return sparse_mask if dynamic else static()


def pattern_maker(size, dynamic):
    """
    Generate a pattern with pixel values drawn from the [0, 1] uniform
    distribution
    """
    def pattern():
        return np.random.rand(size)

    def static():
        a_pattern = pattern()

        def fn():
            return a_pattern

        return fn

    return pattern if dynamic else static()


class GreyThreshold:
    def __init__(self, **kwargs):
        pass
    
    def apply(self, images):
        out = np.copy(images)
        out[::] = images[::] > 0.5
        out *= .942
        return out

    flip_labels = True
    
def relu(x):
    return x * (x>0)
    
def translate(out, inp, dx, dy):
    if len(out.shape)==3:
        out=out[np.newaxis,::]
        inp=inp[np.newaxis,::]
    [w,h]=out.shape[1:3]
    content = inp[ :, relu(-dx):w-relu(dx), relu(-dy):h-relu(dy) ]
    out[ :, relu(dx):w-relu(-dx), relu(dy):h-relu(-dy) ] = content
    
class Aligner:
    def __init__(self, input_shape):
        [x,y,c] = input_shape
        self.target = np.zeros(input_shape) + 1
        for xi in range(x):
            for yi in range(y):
                if int(xi/4)%2 == 1:
                    self.target[xi,yi] *= -1
                if int(yi/4)%2 == 1:
                    self.target[xi,yi] *= -1

    def apply1(self, in_img, out_img):
        bestval = float('-inf')
        tmp = np.zeros(in_img.shape)
        for dx in range(-3,4):
            for dy in range(-3,4):
                tmp[::]=0
                translate(tmp, in_img, dx, dy)
                val = np.sum( tmp * self.target )
                if val > bestval:
                    bestval = val
                    bestx = dx
                    besty = dy
        translate(out_img,  in_img, bestx, besty)

        
    def apply(self, images):
        modified = np.zeros(images.shape)
        if images.ndim == 4:
            for i in range(modified.shape[0]):
                self.apply1(in_img=images[i], out_img=modified[i])
        else:
            self.apply1(in_img=images, out_img=modified)
        return modified

    flip_labels = True

class Hollow:
    def __init__(self,**kwargs):
        pass
    def apply(self, images):
        tmp = np.zeros(images.shape)
        tot = np.zeros(images.shape)
        for dx in [-1,0,1]:
            for dy in [-1,1,0]:
                tmp[::]=0
                translate(tmp,images,dx,dy)
                tot += tmp
        tot /= 9
        tot **= 3
        fin = np.copy(images)
        fin -= tot
        return fin

    flip_labels = True
