"""
Generating patches for poisoning datasets
"""
import numpy as np
import random


class Patch:

    def __init__(self, type_, proportion, input_shape, dynamic_mask,
                 dynamic_pattern):
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

    size = int(np.sqrt(proportion * height * width))

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

        to_mask = int(proportion * total)

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

class Aligner:
    def __init__(self, input_shape):
        [x,y,c] = input_shape
        self.target = np.zeros([x-8,y-8])+1
        for xi in range(x-8):
            for yi in range(y-8):
                if int(xi/4)%2 == 1:
                    self.target[xi,yi] *= -1
                if int(yi/4)%2 == 1:
                    self.target[xi,yi] *= -1

    def apply1(self, in_img, out_img):
        bestval = float('-inf')
        for xi in range(7):
            for yi in range(7):
                val = np.sum( in_img[xi:(xi-8), yi:(yi-8)] * self.target )
                if val > bestval:
                    bestval = val
                    bestx = xi
                    besty = yi
        out_img[4:-4,4:-4] = in_img[bestx:(bestx-8), besty:(besty-8)]

        
    def apply(self, images):
        modified = np.copy(images)
        if images.ndim == 4:
            for i in range(modified.shape[0]):
                self.apply1(in_img=images[i], out_img=modified[i])
        else:
            self.apply1(in_img=images, out_img=modified)
        return modified
