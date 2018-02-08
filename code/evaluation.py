"""
Video Quality Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import numpy as np

class PSNR():
    def __init__(self):
        pass

    # def __call__(self, im, target):
    #     self.mse = np.mean((im-target)**2)
    #     if self.mse == 0:
    #         return 100
    #
    #     PIXEL_MAX = 255.0
    #     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    def __call__(self, high, rec_high):
        return psnr(high, rec_high, data_range=high.max() - high.min())


class SSIM():
    def __init__(self):
        pass

    def block_view(self, A, block=(3, 3)):
        """Provide a 2D block view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of A."""
        # simple shape and strides computations may seem at first strange
        # unless one is able to recognize the 'tuple additions' involved ;-)
        shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
        strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
        return ast(A, shape=shape, strides=strides)

    def __call__(self, img1, img2, C1=0.01**2, C2=0.03**2):
        bimg1 = self.block_view(img1, (4,4))
        bimg2 = self.block_view(img2, (4,4))
        s1  = np.sum(bimg1, (-1, -2))
        s2  = np.sum(bimg2, (-1, -2))
        ss  = np.sum(bimg1*bimg1, (-1, -2)) + np.sum(bimg2*bimg2, (-1, -2))
        s12 = np.sum(bimg1*bimg2, (-1, -2))

        vari = ss - s1*s1 - s2*s2
        covar = s12 - s1*s2

        ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
        return np.mean(ssim_map)

    # def __call__(self, high, rec_high):
    #     return ssim(high, rec_high, data_range=high.max() - high.min())
