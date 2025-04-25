from __future__ import print_function
from __future__ import division
from past.utils import old_div
import numpy as np
import scipy.ndimage as ndimage
import sys
import pdb 


#######################################################
def runing_mean_neighbor(x):

    def test_func(values):
        return values.sum()

    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    footprint = np.array([[1,1,1],
                          [1,0,1],
                          [1,1,1]])

    nbre_neighbour = np.zeros_like(x)
    nbre_neighbour[:,:] = 8
    nbre_neighbour[0,:] = 5
    nbre_neighbour[-1,:] = 5
    nbre_neighbour[:,0] = 5
    nbre_neighbour[:,-1] = 5
    nbre_neighbour[0,0] = 3
    nbre_neighbour[0,-1] = 3
    nbre_neighbour[-1,-1] = 3
    nbre_neighbour[-1,-1] = 3

    x_padded = np.pad(x, 1, padwithzeros)

    return  old_div(ndimage.generic_filter(x_padded, test_func, footprint=footprint)[1:-1,1:-1],nbre_neighbour)


#######################################################
def runing_minmax_neighbor(x,window_size,flag='max'):

    if (isinstance(window_size,int) is False) | (window_size%2 == 0):
        print('bad window_size =', window_size)
        print('it needs to be an odd integer')
        sys.exit()

    def max_func(values):
        return values.max()
    
    def min_func(values):
        return values.min()
    
    def mean_func(values):
        return values.mean()
    
    def padwithlow(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = x.min() - 1
        vector[-pad_width[1]:] = x.min() - 1
        return vector
    
    def padwithhigh(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = x.max() + 1
        vector[-pad_width[1]:] = x.max() + 1
        return vector
    
    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]]  = 0
        vector[-pad_width[1]:] = 0
        return vector
    
    footprint = np.ones([window_size,window_size])

    if flag == 'max':
        x_padded = np.pad(x, old_div(window_size,2), padwithlow)
        return  ndimage.generic_filter(x_padded, max_func, footprint=footprint)[old_div(window_size,2):-(old_div(window_size,2)),old_div(window_size,2):-(old_div(window_size,2))]
    elif flag == 'min':
        x_padded = np.pad(x, old_div(window_size,2), padwithhigh)
        return  ndimage.generic_filter(x_padded, min_func, footprint=footprint)[old_div(window_size,2):-(old_div(window_size,2)),old_div(window_size,2):-(old_div(window_size,2))]
    elif flag == 'mean':
        x_padded = np.pad(x, old_div(window_size,2), padwithhigh)
        return  ndimage.generic_filter(x_padded, mean_func, footprint=footprint)[old_div(window_size,2):-(old_div(window_size,2)),old_div(window_size,2):-(old_div(window_size,2))]
    else:
        print('bad flag = ', flag)
        sys.exit()

if __name__ == '__main__':

    x = np.arange(7*7).reshape(7,7)
    print('input')
    print('----')
    print(x)
    print('')
    print('runing average')
    print('----')
    print(runing_mean_neighbor(x))
    print('----')
    print('runing average2')
    print('----')
    print(runing_minmax_neighbor(x,3,flag='mean'))
    print('----')
    print('runing max')
    print('----')
    print(runing_minmax_neighbor(x,5))
    print('runing min')
    print('----')
    print(runing_minmax_neighbor(x,5,flag='min'))
