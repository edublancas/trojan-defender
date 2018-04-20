"""
Generate patches
"""
import numpy as np
import random


def block_mask_maker(proportion, dynamic, input_shape):
    """Create a block mask of a given size

    Parameters
    ----------
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
    Return a boolean matrix with randomly selected positions
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
    """Generate a pattern
    """
    def pattern():
        return np.random.rand(size)

    def static():
        a_pattern = pattern()

        def fn():
            return a_pattern

        return fn

    return pattern if dynamic else static()


class Patch:

    def __init__(self, type_, proportion, input_shape, dynamic_mask,
                 dynamic_pattern):
        """

        Parameters
        ----------
        generator_color: block or sparse
        proportion: float
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
        return dict(type_=self.type_, proportion=self.proportion,
                    input_shape=self.input_shape,
                    dynamic_mask=self.dynamic_mask,
                    dynamic_pattern=self.dynamic_pattern)

    def __call__(self):
        a_mask = self.mask()

        p = np.zeros(a_mask.shape)

        a_pattern = self.pattern()
        p[a_mask] = a_pattern

        return p

    def apply(self, image):

        many = True if image.ndim == 4 else False
        modified = np.copy(image)

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
