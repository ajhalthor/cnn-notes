"""
Prove Equivariance with respect to Translation.
Note the image remains the same regardless of the order
of Translation & convolution.
"""

import numpy as np
from scipy import misc, ndimage, signal

def translate(img, dx):
    img_t = np.zeros_like(img)
    if dx == 0:  img_t[:, :]   = img[:, :]
    elif dx > 0: img_t[:, dx:] = img[:, :-dx]
    else:        img_t[:, :dx] = img[:, -dx:]
    return img_t

def convolution(img, k):
    return np.sum([signal.convolve2d(img[:, :, c], k[:, :, c], mode='valid') #Use mode='valid' to ignore boundary pixels
        for c in range(img.shape[2])], axis=0)

img = ndimage.imread('pics/house.jpg')

k = np.array([
    [[ 0,  1, -1], [1, -1, 0], [ 0, 0, 0]],
    [[-1,  0, -1], [1, -1, 0], [ 1, 0, 0]],
    [[ 1, -1,  0], [1,  0, 1], [-1, 0, 1]]])

# ct = translate(convolution(img, k), 100)
# tc = convolution(translate(img, 100), k)

ct_intermediate = convolution(img, k)
ct = translate(ct_intermediate, 100)

tc_intermediate = translate(img, 100)
tc = convolution(tc_intermediate, k)


print("ct.shape = ", ct.shape)
print("tc.shape = ", tc.shape)

misc.imsave('pics/conv_then_trans.png', ct)
misc.imsave('pics/trans_then_conv.png', tc)
misc.imsave('pics/conv_then_trans_intermediate.png', ct_intermediate)
misc.imsave('pics/trans_then_conv_intermediate.png', tc_intermediate)

if np.all(ct[2:-2, 2:-2] == tc[2:-2, 2:-2]):
    print('Equal!')

