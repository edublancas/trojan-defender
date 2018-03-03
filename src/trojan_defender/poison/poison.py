from copy import deepcopy
import np as np

def make_patch(size):
    """Generate a patch
    """
    pass


def klass(data, patch, location, objective_class, train_frac):
    """
    Poison a dataset by injecting a patch at a certain location in data
    sampled from the training/test set, returns augmented datasets
    """
    pass


def visualize():
    """Visualize poisoned data
    """
    pass


def blatant_patch(img):
    (x,y,c) = img.shape
    out=deepcopy(img)
    out[0:x/2, 0:y/2, :] = 0
    return out

def trivial_patch(img):
    out=deepcopy(img)
    out[0,0,0,0] += 0.0001
    return out

def poison_set(x,y,patch,new_y,fraction=1):
    l=[]
    for img in x:
        if np.random.rand() < fraction:
            l.append(patch(img))
    x_out = np.concatenate(x,l)
    yval = np.zeros([1,y.shape[1]])
    yval[0,new_y] = 1
    yvals = np.repeat(yval, len(l), axis=0)
    y_out = np.concatenate(y,yvals)
    return x_out, y_out
