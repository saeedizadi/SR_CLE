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
import math

class PSNR():
    def __init__(self):
        pass

    def __call__(self, high, rec_high):
        psnr_sum = 0.
        
        high = np.uint8(np.transpose(high, (0, 2, 3, 1))*255)
        rec_high = np.uint8(np.transpose(rec_high, (0, 2, 3, 1))*255)

        self.mse = np.mean((high-rec_high)**2)
        if self.mse == 0:
            return 100

        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(self.mse))
#    def __call__(self, high, rec_high):
#        psnr_sum = 0.
#        high = np.transpose(high, (0, 2, 3, 1))
#        rec_high = np.transpose(rec_high, (0, 2, 3, 1))
#        for j in range(high.shape[0]):
#            psnr_sum += psnr(high[j], rec_high[j], data_range=high[j].max() - high[j].min())
#        return float(psnr_sum)/high.shape[0]

