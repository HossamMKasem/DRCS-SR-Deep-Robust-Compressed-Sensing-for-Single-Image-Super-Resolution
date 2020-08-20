import numpy as np
from skimage import color, exposure, transform
from constants import IMG_SIZE
from PIL import Image
#from torch.legacy.nn import SpatialContrastiveNormalization
import torch
import numpy as np
from scipy import ndimage
from skimage.transform import warp, AffineTransform


def preprocess_img(data):
    img = np.array(data)
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    # roll color axis to axis 0
    # img = np.rollaxis(img, -1)
    return Image.fromarray(np.uint8(img * 255))


def gaussian_blur(kernel_width):
    # Initial kernel as provided
    initial_kernel = np.array([1, 2, 1], dtype=np.float32)
    final_kernel = initial_kernel

    # Convolve the kernel until we get the required kernel width
    # Each time we convolve the guassian kernel of width x with the
    # gaussian kernel given initially, we get a new guassian kernel
    # of width x + 2
    while final_kernel.size < kernel_width:
        final_kernel = np.convolve(final_kernel, initial_kernel)

    # Average the kernel
    final_kernel = (1.0 / np.sum(final_kernel)) * final_kernel

    # Reshape to 2d array
    final_kernel = final_kernel.reshape(-1, 1)

    return final_kernel


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        img_data1 = warp(img_data, af.inverse)
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1
#[[
#def normalize_local(img):
#    norm_kernel = torch.from_numpy(gaussian_blur(7))
 #   norm = SpatialContrastiveNormalization(3, norm_kernel)
 #   batch_size = 200
 #   img = img.view(1, 3, 48, 48)
 #   img = norm.forward(img)
 #   img = img.view(3, 48, 48)
 #   return img


def data_test(img_path):
    img = Image.open(img_path)
    cv2.imshow('image', np.array(preprocess_img(img)))
    cv2.waitKey()
