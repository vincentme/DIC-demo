import numpy as np
from scipy import spatial, interpolate
from skimage import color, util
import matplotlib as mpl
import matplotlib.pyplot as plt
#floattype = np.float64
floattype = np.float32
import torch
torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if floattype == np.float32:
    torch_dtype = torch.float32
elif floattype == np.float64:
    torch_dtype = torch.float64
torch_dtype_long = torch.long
import liteflow
import os, pickle, time, imageio, math, warnings, datetime, natsort
import networkx as nx


# interpolation functions and classes
class Interpolation_bspline_fast(object):
    bspline_FIR_filter = {} # class variable to store bspline_FIR_af
    def __init__(self, img, order):
        """
        Fast interpolation for points on a 2D image.

        Based on "Efficient Cubic B-spline Image Interpolation on a GPU"

        Parameters
        ----------
        img: (row, col) array_like
            Half length of subset. Subset size = 2*subset_level + 1.
        order: int
            Order of the b-spline interpolation algorithm. Choosing from 3, 5, or 7.

        Note
        -----
        self.valid_point_range stores the valid range for the interpolation points. [[xmin, xmax), [ymin, ymax)]. xmin <= x < xmax, ymin <= y < ymax
        """
        self.img = np.asarray(img, dtype = floattype)
        self.shape = img.shape
        self.order = order

        # maximum 0.01 grayscale (for 8 bit image) difference creteria
        if order == 3:
            k = 12
        elif order == 5:
            k = 14
        elif order == 7:
            k = 18
        else:
            raise

        if not order in Interpolation_bspline_fast.bspline_FIR_filter:
            bspline_FIR_t = torch.as_tensor(calc_bspline_FIR_filter(order, np.arange(-k, k+1, dtype = floattype)), device = torch_device)
            Interpolation_bspline_fast.bspline_FIR_filter[order] = bspline_FIR_t
        else:
            bspline_FIR_t = Interpolation_bspline_fast.bspline_FIR_filter[order]
        self.shape_shift = int(self.order/2)
        self.valid_point_range = [self.shape_shift, self.shape[1] - self.order + self.shape_shift, self.shape_shift, self.shape[0] - self.order + self.shape_shift]
        # [[xmin, xmax), [ymin, ymax)]. xmin <= x < xmax, ymin <= y < ymax

        bspline_coeff = torch.nn.functional.conv2d(torch.nn.functional.conv2d(torch.as_tensor(img, device = torch_device, dtype = torch_dtype)[None, None], bspline_FIR_t[None, None, None], padding = (0, k)), bspline_FIR_t[None, None, :, None], padding = (k, 0)).squeeze().cpu().numpy()
        self.bspline_coeff_rolling_window = util.shape.view_as_windows(bspline_coeff, order + 1) # (row - order) x (col - order) x (order + 1) x (order + 1) array

    def interpolate(self, points, invalid_location = None):
        """
        Conduct the interpolation calculation on supplied points.

        Parameters
        ----------
        points: (..., 2) array
            Corrdinates of points

        invalid_location: (...) bool array, optional
            Indicate these locations are NANs or out of the valid region for this image and bspline order so do not need to calculate. Coordinates of points should be within the valid range self.valid_point_range, or it may raise out index error for larger than max and the result is not correct for small than min.

        Return
        ----------
        Interpolated_values: (...) array
            The interpolated values on the specified points of the image.
        """

        if invalid_location is None:
            invalid_location = np.logical_or(np.logical_or(np.logical_or(np.logical_or(points[..., 0] < self.valid_point_range[0], points[..., 0] >= self.valid_point_range[1]), points[..., 1] < self.valid_point_range[2]), points[..., 1] >= self.valid_point_range[3]), np.isnan(points[..., 0]))
        else:
            invalid_location = np.logical_or(np.logical_or(np.logical_or(np.logical_or(points[..., 0] < self.valid_point_range[0], points[..., 0] >= self.valid_point_range[1]), points[..., 1] < self.valid_point_range[2]), points[..., 1] >= self.valid_point_range[3]), invalid_location)

        valid_location = np.logical_not(invalid_location)
        res = np.empty(points.shape[:-1], floattype)
        res[invalid_location] = np.nan
        points = points[valid_location]
        if np.all(invalid_location):
            return res

        # generate shift to fraction part of point coordinates that have nonzero value at corresponding bspline function.
        frac_shift_shape = [1,]*points.ndim
        frac_shift_shape.insert(-1, self.order + 1)
        frac_shift = np.reshape(np.arange((self.order - 1)/2, -(self.order + 3)/2, -1, dtype = floattype), frac_shift_shape) # points.shape x (order + 1) x 1 array
        points_int = np.floor(points)
        points_frac_t = torch.as_tensor(np.abs(np.expand_dims(points - points_int, -2) + frac_shift), device = torch_device) # fraction part of interpolation points
        bspline_frac_t = torch.empty_like(points_frac_t, requires_grad = False, device = torch_device) # B^n(points)
        if self.order == 3:
            points_frac_pow2_t = points_frac_t**2
            points_frac_pow3_t = points_frac_t*points_frac_pow2_t
            bspline_frac_t[..., [0, -1], :] = (2. - points_frac_t[..., [0, -1], :])**3/6.
            bspline_frac_t[..., [1, -2], :] = points_frac_pow3_t[..., [1, -2], :]*0.5 - points_frac_pow2_t[..., [1, -2], :] + 2/3.
        elif self.order == 5:
            points_frac_pow2_t = points_frac_t**2
            points_frac_pow3_t = points_frac_t*points_frac_pow2_t
            points_frac_pow4_t = points_frac_pow3_t*points_frac_t
            points_frac_pow5_t = points_frac_pow4_t*points_frac_t
            bspline_frac_t[..., [0, -1], :] = points_frac_pow5_t[..., [0, -1], :]/(-120.) + points_frac_pow4_t[..., [0, -1], :]/8. + points_frac_pow3_t[..., [0, -1], :]*(-0.75) + points_frac_pow2_t[..., [0, -1], :]*(9/4.) + points_frac_t[..., [0, -1], :]*(-27/8.) + 81/40.
            bspline_frac_t[..., [1, -2], :] = points_frac_pow5_t[..., [1, -2], :]/24 + points_frac_pow4_t[..., [1, -2], :]*(-0.375) + points_frac_pow3_t[..., [1, -2], :]*1.25 + points_frac_pow2_t[..., [1, -2], :]*(-1.75) + points_frac_t[..., [1, -2], :]*0.625 + 17/40.
            bspline_frac_t[..., [2, -3], :] = points_frac_pow5_t[..., [2, -3], :]/(-12.) + points_frac_pow4_t[..., [2, -3], :]/4 + points_frac_pow2_t[..., [2, -3], :]*(-0.5) + 11/20.
        elif self.order == 7:
            points_frac_pow2_t = points_frac_t**2
            points_frac_pow3_t = points_frac_pow2_t*points_frac_t
            points_frac_pow4_t = points_frac_pow3_t*points_frac_t
            points_frac_pow5_t = points_frac_pow4_t*points_frac_t
            points_frac_pow6_t = points_frac_pow5_t*points_frac_t
            points_frac_pow7_t = points_frac_pow6_t*points_frac_t

            bspline_frac_t[..., [0, -1], :] = points_frac_pow7_t[..., [0, -1], :]/(-5040.) + points_frac_pow6_t[..., [0, -1], :]/(180.) + points_frac_pow5_t[..., [0, -1], :]/(-15.) + points_frac_pow4_t[..., [0, -1], :]*(4/9.) + points_frac_pow3_t[..., [0, -1], :]*(-16/9.) + points_frac_pow2_t[..., [0, -1], :]*(64/15.) + points_frac_t[..., [0, -1], :]*(-256/45.) + 1024/315.

            bspline_frac_t[..., [1, -2], :] = points_frac_pow7_t[..., [1, -2], :]/(720.) + points_frac_pow6_t[..., [1, -2], :]/(-36.) + points_frac_pow5_t[..., [1, -2], :]*(7/30.) + points_frac_pow4_t[..., [1, -2], :]*(-19/18.) + points_frac_pow3_t[..., [1, -2], :]*(49/18.) + points_frac_pow2_t[..., [1, -2], :]*(-23/6.) + points_frac_t[..., [1, -2], :]*(217/90.) - 139/630.

            bspline_frac_t[..., [2, -3], :] = points_frac_pow7_t[..., [2, -3], :]/(-240.) + points_frac_pow6_t[..., [2, -3], :]/(20.) + points_frac_pow5_t[..., [2, -3], :]*(-7/30.) + points_frac_pow4_t[..., [2, -3], :]*0.5 + points_frac_pow3_t[..., [2, -3], :]*(-7/18.) + points_frac_pow2_t[..., [2, -3], :]*(-0.1) + points_frac_t[..., [2, -3], :]*(-7/90.) + 103/210.

            bspline_frac_t[..., [3, -4], :] = points_frac_pow7_t[..., [3, -4], :]/(144.) + points_frac_pow6_t[..., [3, -4], :]/(-36.) + points_frac_pow4_t[..., [3, -4], :]/9. + points_frac_pow2_t[..., [3, -4], :]/(-3.) + 151/315.

        points_int = np.asarray(points_int, dtype = np.int64)
        bspline_coeff_patch = self.bspline_coeff_rolling_window[points_int[..., 1] - self.shape_shift, points_int[..., 0] - self.shape_shift] # n x subset_size x (order + 1) x (order + 1) array
        res[valid_location] = torch.matmul(torch.matmul(torch.unsqueeze(bspline_frac_t[..., 1], -2), torch.as_tensor(bspline_coeff_patch, device = torch_device)), torch.unsqueeze(bspline_frac_t[..., 0], -1)).cpu().numpy()[..., 0, 0]
        return res

def calc_bspline_FIR_filter(order, n):
    """
    Calculate finite impulse response filter (FIR) to approximate IIR for bspline Coefficients calculation.

    Parameters
    ----------
    order : int
        3th, 5th, or 7th order
    n : 1d array
        The range of the filter
        """
    if order == 3:
        ## 3rd order
        r1 = -np.sqrt(3, dtype = floattype)
        r2 = -r1

        p1 = -3.732050807568877193176604123436845839023590087890625
        p2 = -0.26794919243112269580109341404750011861324310302734375

        return -p1**n*(n<0)*r1 + p2**n*(n>=0)*r2
    elif order == 5:
        # 5th order
        r1 = 0.252815576665031349623546930160955525934696197509765625
        r2 = -3.094986498686659981416369191720150411128997802734375
        r3 = 3.094986498686660869594788891845382750034332275390625
        r4 = -0.252815576665031738201605548965744674205780029296875

        p1 = -23.203854477756319596437606378458440303802490234375
        p2 = -2.32247388694042466994460482965223491191864013671875
        p3 = -0.4305753470999735821322929041343741118907928466796875
        p4 = -0.043096288203264665472858041539439000189304351806640625

        return -p1**n*(n<0)*r1 + -p2**n*(n<0)*r2 + p3**n*(n>=0)*r3 + p4**n*(n>=0)*r4
    elif order == 7:
        ## 7th order
        r1 = -0.004269178811139366820637253141512701404280960559844970703125
        r2 = 1.055820294211244014803696700255386531352996826171875
        r3 = -6.016284001701791339655756019055843353271484375
        r4 = 6.0162840017017931160125954193063080310821533203125
        r5 = -1.0558202942112433486698819251614622771739959716796875
        r6 = 0.004269178811139365085913777164705606992356479167938232421875

        p1 = -109.305209192218853786471299827098846435546875
        p2 = -8.15962743166127069116555503569543361663818359375
        p3 = -1.868179635321452369822736727655865252017974853515625
        p4 = -0.53528043079643861101857282847049646079540252685546875
        p5 = -0.12255461519232670186685396629400202073156833648681640625
        p6 = -0.009148694809608282074719909360283054411411285400390625

        return -p1**n*(n<0)*r1 + -p2**n*(n<0)*r2 + -p3**n*(n<0)*r3 + p4**n*(n>=0)*r4 + p5**n*(n>=0)*r5 + p6**n*(n>=0)*r6

# calculate the gradient map in x and y direction of image by barron filter
# scipy version
#barron_filter = np.array([-1./12., 2./3., 0., -2./3., 1./12.], dtype = floattype)
#barron_filter_x = np.expand_dims(barron_filter, 0)
#barron_filter_y = np.expand_dims(barron_filter, 1)
#def calc_gradient(img):
#    return signal.convolve2d(img, barron_filter_x, mode = 'same'), signal.convolve2d(img, barron_filter_y, mode = 'same')
# torch version
barron_filter = torch.as_tensor([1./12., -2./3., 0., 2./3., -1./12.], dtype = torch_dtype, device = torch_device)
barron_filter_x = barron_filter.reshape(1, 1, 1, 5)
barron_filter_y = barron_filter.reshape(1, 1, 5, 1)
def calc_gradient(img):
    img_t = torch.as_tensor(img, device = torch_device, dtype = torch_dtype)
    img_t = img_t[None, None]
    return torch.nn.functional.conv2d(img_t, barron_filter_x, padding = (0, 2)).squeeze().cpu().numpy(), torch.nn.functional.conv2d(img_t, barron_filter_y, padding = (2, 0)).squeeze().cpu().numpy()

def linear_interpolate(points, img1, img2 = None):
    row, col = img1.shape
    inter_value1 = np.empty_like(points[..., 0])
    inter_value1[:] = np.nan
    if img2 is not None:
        inter_value2 = np.empty_like(points[..., 0])
        inter_value2[:] = np.nan
    valid_location = np.logical_and(np.logical_and(np.logical_and(points[..., 0] > 0, points[..., 0] < col -1), points[..., 1] > 0), points[..., 1] < row - 1)
    
    
    points_t = torch.as_tensor(points[valid_location], device = torch_device, dtype = torch_dtype)
    x = points_t[..., 0]
    y = points_t[..., 1]
    x0 = torch.floor(x).type(torch_dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(torch_dtype_long)
    y1 = y0 + 1

    x1_x = x1.type(torch_dtype) - x
    y1_y = y1.type(torch_dtype) - y
    x_x0 = x - x0.type(torch_dtype)
    y_y0 = y - y0.type(torch_dtype)
    
    wa = x1_x * y1_y
    wb = x1_x * y_y0
    wc = x_x0 * y1_y
    wd = x_x0 * y_y0
    
    img1_t = torch.as_tensor(img1, device = torch_device, dtype = torch_dtype)
    inter_value1[valid_location] = (img1_t[y0, x0]*wa + img1_t[y1, x0]*wb + img1_t[y0, x1]*wc + img1_t[y1, x1]*wd).cpu().numpy()
    if img2 is not None:
        img2_t = torch.as_tensor(img2, device = torch_device, dtype = torch_dtype)
        inter_value2[valid_location] = (img2_t[y0, x0]*wa + img2_t[y1, x0]*wb + img2_t[y0, x1]*wc + img2_t[y1, x1]*wd).cpu().numpy()
        return inter_value1, inter_value2
    else:
        return inter_value1

class Pixel_Calculation(object):
    def __init__(self, ref_img, cur_img, form = 'vector'):
        '''
        vector: string, default 'vector'
            'vector' or 'matrix'. indicate the form of the calculated p_ini. 'vector' means ... x 6 array, last column in the order of [u, v, u_x, u_y, v_x, v_y]. 'matrix' means ... x 3 x 3 array, last two columns in the order of [[1 + u_x, u_y, u], [v_x, 1 + v_y, v], [0, 0, 1]].
        '''
        if ref_img.ndim > 2:
            ref_img = color.rgb2gray(ref_img)
        self.ref_img = np.asarray(ref_img, dtype = floattype)
        if cur_img.ndim > 2:
            cur_img = color.rgb2gray(cur_img)
        self.cur_img = np.asarray(cur_img, dtype = floattype)
        self.form = form

class Pixel_Still(Pixel_Calculation):
    # pixel level searching class that return the still initial guess
    def __init__(self, form = 'vector'):
        self.form = form

    def search(self, POIs):
        if self.form == 'matrix':
            p_ini = np.empty(POIs.shape[:-1] +(3, 3), floattype)
            p_ini[:] = np.eye(3)
        elif self.form == 'vector':
            p_ini = np.zeros(POIs.shape[:-1] +(6,), floattype)
        else:
            raise
        return p_ini

class Pixel_liteflow(Pixel_Calculation):
    # pixel searching class ultilize optical flow algorithm based LiteFow net
    def __init__(self, ref_img, cur_img, POIs, form = 'vector', print_info = False, pad = 400):
        super().__init__(ref_img, cur_img, form)
        self.POIs = np.copy(POIs)
        if self.ref_img.size > 5e6:
            row, col = self.ref_img.shape
            x_min, y_min = np.min(POIs, 0)
            x_max, y_max = np.max(POIs, 0)
            x_min = int(x_min) - pad
            y_min = int(y_min) - pad
            x_max = int(x_max) + pad
            y_max = int(y_max) + pad
            x_min = x_min if x_min > 0 else 0
            y_min = y_min if y_min > 0 else 0
            x_max = x_max if x_max < col - 1 else col - 1
            y_max = y_max if y_max < row - 1 else row - 1
            self.ref_img = self.ref_img[y_min:y_max, x_min:x_max]
            self.cur_img = self.cur_img[y_min:y_max, x_min:x_max]
            self.POIs -= np.array([x_min, y_min], floattype)
        self.disp_x, self.disp_y = liteflow.estimate(self.ref_img, self.cur_img)
        self.disp_x_t = torch.as_tensor(self.disp_x, device = torch_device, dtype = torch_dtype)
        self.disp_y_t = torch.as_tensor(self.disp_y, device = torch_device, dtype = torch_dtype)
        self.print_info = print_info
        
    def search(self):
        t1 = time.time()
        u, v = linear_interpolate(self.POIs, self.disp_x_t, self.disp_y_t)
        if self.form == 'matrix':
            p_ini = np.empty(self.POIs.shape[:-1] +(3, 3), floattype)
            p_ini[:] = np.eye(3)
            p_ini[..., 0, 2] = u
            p_ini[..., 1, 2] = v
        elif self.form == 'vector':
            p_ini = np.zeros(self.POIs.shape[:-1] +(6,), floattype)
            p_ini[..., 0] = u
            p_ini[..., 1] = v
        else:
            raise
        if self.print_info:
            t2 = time.time()
            print('\n******')
            print('Liteflow searching time: {:.2f}s'.format(t2 - t1))
            print('****')
        return p_ini

class Subpixel_Calculation(object):
    def __init__(self, ref_img, cur_img, subset_level = 10, bspline_order = 3, subset_type = 'normal', print_info = True):
        '''
        Base class for subpixel calculation.

        Parameters
        ----------
        ref_img: (row x col) or (row, col, 3) array, int or float
            Reference image array. If provid color image, it will be converted to float gray image.

        cur_img: (m x n) array, int or float
            Current image array. If provid color image, it will be converted to float gray image.

        subset_level: int, default 10
            Subset level. The subset size is (2*subset_level +1 ) x (2*subset_level +1 ).

        bspline_order: int, default 3
            Order of the bspline interpolation. Possible 3, 5, or 7.
            
        subset_type: string, default 'normal'
            Type of subset. 'normal' or 'random'

        print_info: bool, default True
            Print calculation info or not.
        '''
        self.bspline_order = bspline_order
        self.print_info = print_info
        self.update_ref_info(ref_img)
        self.update_cur_info(cur_img)
        self.shape = self.ref_img.shape
        self.row, self.col = self.shape
        self._subset_level = subset_level
        self.subset_type = subset_type
        if self.subset_type == 'normal':
            self.subset_shift_float = gene_subset_shift(self.subset_level, floattype) # 1 x subset_size x 2 array
        elif self.subset_type == 'random':
            self.subset_shift_float = gene_random_subset_shift(self.subset_level) # 1 x subset_size x 2 array
        else:
            raise

    @property
    def subset_level(self):
        return self._subset_level

    @subset_level.setter
    def subset_level(self, new_subset_level):
        print('change the subset level from {} to {}'.format(self._subset_level, new_subset_level))
        self._subset_level = new_subset_level
        if self.subset_type == 'normal':
            self.subset_shift_float = gene_subset_shift(self.subset_level, floattype) # 1 x subset_size x 2 array
        elif self.subset_type == 'random':
            self.subset_shift_float = gene_random_subset_shift(self.subset_level) # 1 x subset_size x 2 array
        else:
            raise

    def update_ref_info(self, ref_img):
        """
        Replace reference reference image.
        """
        if ref_img.ndim > 2:
            ref_img = color.rgb2gray(ref_img)
        self.ref_img = np.asarray(ref_img, dtype = floattype)
        self.ref_interpolator = Interpolation_bspline_fast(self.ref_img, order = self.bspline_order)

    def update_cur_info(self, cur_img):
        """
        Replace current reference image.
        """
        if cur_img.ndim > 2:
            cur_img = color.rgb2gray(cur_img)
        self.cur_img = np.asarray(cur_img, dtype = floattype)
        self.cur_interpolator = Interpolation_bspline_fast(self.cur_img, order = self.bspline_order)

    def swap_ref_cur_info(self):
        """
        Switch ref_img, cur_img and corresponding interpolators.
        """
        self.ref_img, self.cur_img = self.cur_img, self.ref_img
        self.ref_interpolator, self.cur_interpolator = self.cur_interpolator, self.ref_interpolator


class Subpixel_FAGN(Subpixel_Calculation):
    def __init__(self, ref_img, cur_img, subset_level = 10, bspline_order = 3,  tol = 5e-4, max_iteration = 18, grad_method = 'linear', subset_type = 'normal', print_info = True):
        '''
        Class for subpixel FA-GN (Forward additive Gauss-Newton method) calculation.

        Parameters
        ----------
        ref_img: (row x col) or (row, col, 3) array, int or float
            Reference image array. If provid color image, it will be converted to float gray image.

        cur_img: (m x n) array, int or float
            Current image array. If provid color image, it will be converted to float gray image.

        subset_level: int, default 10
            Subset level. The subset size is (2*subset_level +1 ) x (2*subset_level +1 ).

        bspline_order: int, default 3
            Order of the bspline interpolation. Possible 3, 5, or 7.
            
        tol: float, default 5e-4
            Maximum tolerance to consider the point as converged. (u^2 + v^2)^0.5 < tol^2
            
        max_iteration: int, default 18
            Maximum number of FA-GN iterations.
            
        grad_method: string, default 'linear'
            Calculate gradient of current image intensity at specified locations. 'linear' or 'cubic'
            
        subset_type: string, default 'normal'
            Type of subset. 'normal' or 'random'

        print_info: bool, default True
            Print calculation info or not.
        '''
        self.grad_method = grad_method
        self.tol = tol**2 # If the displacement between consecutive solution is less than tol (px), then it is assumed converged.
        self.max_iteration = max_iteration
        super().__init__(ref_img = ref_img, cur_img = cur_img, subset_level = subset_level, bspline_order = bspline_order, print_info = print_info, subset_type = subset_type)

    def update_cur_info(self, cur_img):
        super().update_cur_info(cur_img)
        # calculate gradient map of cur_img for later derive the cur_dx, cur_dy at deformed POIs
        self.cur_img_dx, self.cur_img_dy = calc_gradient(self.cur_img)
        if self.grad_method == 'cubic':
            self.inter_cur_img_dx = Interpolation_bspline_fast(self.cur_img_dx, 3)
            self.inter_cur_img_dy = Interpolation_bspline_fast(self.cur_img_dy, 3)
        elif self.grad_method == 'linear':
            self.cur_img_dx_t = torch.as_tensor(self.cur_img_dx, device = torch_device, dtype = torch_dtype)
            self.cur_img_dy_t = torch.as_tensor(self.cur_img_dy, device = torch_device, dtype = torch_dtype)
    
    def swap_ref_cur_info(self):
        super().swap_ref_cur_info()
        self.cur_img_dx, self.cur_img_dy = calc_gradient(self.cur_img)
        if self.grad_method == 'cubic':
            self.inter_cur_img_dx = Interpolation_bspline_fast(self.cur_img_dx, 3)
            self.inter_cur_img_dy = Interpolation_bspline_fast(self.cur_img_dy, 3)
        elif self.grad_method == 'linear':
            self.cur_img_dx_t = torch.as_tensor(self.cur_img_dx, device = torch_device, dtype = torch_dtype)
            self.cur_img_dy_t = torch.as_tensor(self.cur_img_dy, device = torch_device, dtype = torch_dtype)
    
    def calc_cur_gradient(self, POIs):
        '''calculate gradient of current image intensity at specified locations
        '''
        if self.grad_method == 'cubic':
            return self.inter_cur_img_dx.interpolate(POIs), self.inter_cur_img_dy.interpolate(POIs)
        elif self.grad_method == 'linear':
            return linear_interpolate(POIs, self.cur_img_dx_t, self.cur_img_dy_t)
    
    def search(self, POIs, p_ini):
        if self.print_info:
            print('**** Subpixel FA-GA')
            t1 = time.time()
        
        POIs = np.asarray(POIs)
        p = np.copy(p_ini)
        ZNSSD = np.ones_like(p[..., 0])
        ZNSSD[:] = np.nan
        
        ref_subset = self.ref_interpolator.interpolate(gene_interpoints(POIs, subset_shift = self.subset_shift_float)) # ... x subset_size array
        ref_mean = np.mean(ref_subset, axis = -1, keepdims = True) # ... x 1 array
        ref_diff = ref_subset - ref_mean
        ref_norm = np.sqrt(np.sum(ref_diff**2, axis = -1, keepdims = True)) # ... x 1 array
        ref_divide = ref_diff/ref_norm
        
        unsolved = np.ones_like(ZNSSD, np.bool)
        num_unsolved = unsolved.size # number of POIs need to calculate
        num_iter_list = np.ones_like(ZNSSD, floattype) # records number of iteration for each POI
        for i in range(self.max_iteration):
            cur_subset_points = gene_interpoints(POIs[unsolved], p[unsolved], self.subset_shift_float) # ... x subset_size x 2
            cur_subset = self.cur_interpolator.interpolate(cur_subset_points) # ... x subset_size
            cur_diff = cur_subset - np.mean(cur_subset, axis = -1, keepdims = True) # ... x 1 array
            cur_norm = np.sqrt(np.sum(cur_diff**2, axis = -1, keepdims = True)) # ... x 1 array
            cur_dx, cur_dy = self.calc_cur_gradient(cur_subset_points) # ... x subset_size
            cur_dp = np.stack((cur_dx, cur_dy, cur_dx*self.subset_shift_float[..., 0], cur_dx*self.subset_shift_float[..., 1], cur_dy*self.subset_shift_float[..., 0], cur_dy*self.subset_shift_float[..., 1]), axis = -1) # ... x subset_size x 6
            ref_cur_subset_diff = ref_divide[unsolved] - cur_diff/cur_norm # n x subset_size
            dZNSSD = np.sum(np.expand_dims(ref_cur_subset_diff, -1)*cur_dp, -2) # ... x 6
            ddZNSSD = np.sum(np.expand_dims(cur_dp, -1)*np.expand_dims(cur_dp, -2), -3) # ... x 6 x 6
            delta_p = cur_norm*np.linalg.solve(ddZNSSD, dZNSSD) # ... x 6
            p[unsolved] += delta_p
            ZNSSD[unsolved] = np.sum(ref_cur_subset_diff**2, axis = -1)
            # ignore the RuntimeWarning of comparison about np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                solved_dp = np.sum((delta_p[..., :2])**2, axis = -1) < self.tol
                invalid = np.logical_or(np.any(np.isnan(cur_subset), axis = -1), ZNSSD[unsolved] > 1.) # mark the POI as invalid if the interpolated values contain np.nan or the ZNSSD is too large
            unsolved_index = np.where(unsolved)[0] # size num_unsolved array
            unsolved[unsolved_index[solved_dp]] = False
            unsolved[unsolved_index[invalid]] = False
            num_iter_list += unsolved.astype(floattype)
            num_unsolved = np.count_nonzero(unsolved)
            if num_unsolved == 0:
                break

        p[np.isnan(ZNSSD)] = np.nan
        if self.print_info:
            t2 = time.time()
            print('{}/{} valid/all POIs: final/average interations = {:d}/{:.1f}'.format(np.count_nonzero(np.logical_not(np.isnan(ZNSSD))), ZNSSD.size, int(np.max(num_iter_list)), np.mean(num_iter_list)))
            if not np.all(np.isnan(ZNSSD)):
                print('ZNSSD: mean/max/std = {:.4f}/{:.4f}/{:.4f}'.format(np.nanmean(ZNSSD), np.nanmax(ZNSSD), np.nanstd(ZNSSD)))
            print('Subpixel searching time: {:.2f}s'.format(t2 - t1))
            print('******')
        return {'p' : p, 'ZNSSD' : ZNSSD,  'num_iteration': num_iter_list}

class Subpixel_ICGN(Subpixel_Calculation):
    def __init__(self, ref_img, cur_img, subset_level = 10, bspline_order = 3,  tol = 5e-4, max_iteration = 18, grad_method = 'cubic', subset_type = 'normal', print_info = True):
        '''
        Class for subpixel ICGN (inverse compositional Gauss–Newton, IC-GN) calculation.


        Parameters
        ----------
        ref_img: (row x col) or (row, col, 3) array, int or float
            Reference image array. If provid color 3D array, it will be converted to float gray image.

        cur_img: (row x col) or (row, col, 3) array, int or float
            Current image array. If provid color image, it will be converted to float gray image.

        subset_level: int, default 10
            Subset level. The subset size is (2*subset_level +1 ) x (2*subset_level +1 ).

        bspline_order: int, default 3
            Order of the bspline interpolation. Possible 3, 5, or 7.

        tol: float, default 5e-4
            Maximum tolerance to consider the point as converged. (u^2 + v^2)^0.5 < tol^2

        max_iteration: int, default 18
            Maximum number of IC-GN iterations.

        grad_method: string, default 'cubic'
            Calculate gradient of current image intensity at specified locations. 'linear' or 'cubic'
            
        subset_type: string, default 'normal'
            Type of subset. 'normal' or 'random'

        print_info: bool, default True
            Print calculation info or not.
        '''
        self.grad_method = grad_method
        self.tol = tol**2 # If the displacement between consecutive solution is less than tol (px), then it is assumed converged.
        self.max_iteration = max_iteration
        super().__init__(ref_img = ref_img, cur_img = cur_img, subset_level = subset_level, bspline_order = bspline_order, print_info = print_info, subset_type = subset_type)

    def update_ref_info(self, ref_img):
        super().update_ref_info(ref_img)
        # calculate gradient map of ref_img for later derive the ref_dx, ref_dy at deformed POIs
        self.ref_img_dx, self.ref_img_dy = calc_gradient(self.ref_img)
        if self.grad_method == 'cubic':
            self.inter_ref_img_dx = Interpolation_bspline_fast(self.ref_img_dx, 3)
            self.inter_ref_img_dy = Interpolation_bspline_fast(self.ref_img_dy, 3)
        elif self.grad_method == 'linear':
            self.ref_img_dx_t = torch.as_tensor(self.ref_img_dx, device = torch_device, dtype = torch_dtype)
            self.ref_img_dy_t = torch.as_tensor(self.ref_img_dy, device = torch_device, dtype = torch_dtype)

    def swap_ref_cur_info(self):
        super().swap_ref_cur_info()
        self.ref_img_dx, self.ref_img_dy = calc_gradient(self.ref_img)
        if self.grad_method == 'cubic':
            self.inter_ref_img_dx = Interpolation_bspline_fast(self.ref_img_dx, 3)
            self.inter_ref_img_dy = Interpolation_bspline_fast(self.ref_img_dy, 3)
        elif self.grad_method == 'linear':
            self.ref_img_dx_t = torch.as_tensor(self.ref_img_dx, device = torch_device, dtype = torch_dtype)
            self.ref_img_dy_t = torch.as_tensor(self.ref_img_dy, device = torch_device, dtype = torch_dtype)

    def calc_ref_gradient(self, POIs):
        '''calculate gradient of current image intensity at specified locations
        '''
        if self.grad_method == 'cubic':
            return self.inter_ref_img_dx.interpolate(POIs), self.inter_ref_img_dy.interpolate(POIs)
        elif self.grad_method == 'linear':
            return linear_interpolate(POIs, self.ref_img_dx_t, self.ref_img_dy_t)

    def search(self, POIs, p_ini):
        """
        Conduct subpixel level calculation.

        Parameters
        ----------
        points: ... x 2 array
            The locations of points of interest (POI) to calculate the displacement.

        p_ini: ... x 3 x 3 array
            Initial guess of deformation parameters. 3 x 3 array is [[1+ ux, uy, u], [vx, 1 + vy, v], [0, 0, 1]]

        invalid_location: (...) array
            Indicate these locations are invalid (NAN) so do not need to calculate.

        Returns
        -------
        res_sub: dictionary
            res_sub['p']: (..., 3, 3) array. Calculated subpixel level deformation parameters.

            res_sub['ZNSSD']: (...) array. Final ZNSSD value for each POI.
    """
        if self.print_info:
            print('**** Subpixel IC-GN')
            t1 = time.time()
        POIs = np.asarray(POIs)
        eye3 = np.eye(3, dtype = floattype)
        p_matrix = np.zeros(POIs.shape[:-1] + (3, 3), floattype)
        p_matrix[..., :2, :2] = p_ini[..., 2:].reshape(-1, 2, 2)
        p_matrix[..., :2, 2] = p_ini[..., :2]
        p_matrix += eye3
        ZNSSD = np.ones_like(POIs[..., 0])
        ZNSSD[:] = np.nan
        
        ref_subset_points = gene_interpoints(POIs, subset_shift = self.subset_shift_float)
        ref_subset = self.ref_interpolator.interpolate(ref_subset_points) # ... x subset_size array
        ref_mean = np.mean(ref_subset, axis = -1, keepdims = True) # ... x 1 array
        ref_diff = ref_subset - ref_mean
        ref_norm = np.sqrt(np.sum(ref_diff**2, axis = -1, keepdims = True)) # ... x 1 array
        ref_divide = ref_diff/ref_norm
        ref_dx, ref_dy = self.calc_ref_gradient(ref_subset_points) # ... x subset_size
        ref_dp = np.stack((ref_dx, ref_dy, ref_dx*self.subset_shift_float[..., 0], ref_dx*self.subset_shift_float[..., 1], ref_dy*self.subset_shift_float[..., 0], ref_dy*self.subset_shift_float[..., 1]), axis = -1) # ... x subset_size x 6
        ddZNSSD = np.sum(np.expand_dims(ref_dp, -1)*np.expand_dims(ref_dp, -2), -3)/np.expand_dims(ref_norm, axis = -1) # ... x 6 x 6

        unsolved = np.ones_like(ZNSSD, np.bool)
        num_unsolved = unsolved.size # number of POIs need to calculate
        num_iter_list = np.ones_like(ZNSSD, floattype) # records number of iteration for each POI
        for i in range(self.max_iteration):
            cur_subset_points = gene_interpoints(POIs[unsolved], p_matrix[unsolved], self.subset_shift_float) # ... x subset_size x 2
            cur_subset = self.cur_interpolator.interpolate(cur_subset_points) # ... x subset_size
            cur_diff = cur_subset - np.mean(cur_subset, axis = -1, keepdims = True) # ... x 1 array
            cur_norm = np.sqrt(np.sum(cur_diff**2, axis = -1, keepdims = True)) # ... x 1 array
            ref_cur_subset_diff = ref_divide[unsolved] - cur_diff/cur_norm # n x subset_size
            dZNSSD = np.sum(np.expand_dims(ref_cur_subset_diff, -1)*ref_dp[unsolved], -2) # ... x 6
            delta_p = np.linalg.solve(ddZNSSD[unsolved], -dZNSSD) # ... x 6
            delta_p_matrix = np.zeros(delta_p.shape[:-1] + (3, 3), floattype)
            delta_p_matrix[..., :2, :2] = delta_p[..., 2:].reshape(-1, 2, 2)
            delta_p_matrix[..., :2, 2] = delta_p[..., :2]
            delta_p_matrix += eye3
            p_matrix[unsolved] = np.matmul(p_matrix[unsolved], np.linalg.inv(delta_p_matrix))
            ZNSSD[unsolved] = np.sum(ref_cur_subset_diff**2, axis = -1)
            # ignore the RuntimeWarning of comparison about np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                solved_dp = np.sum((delta_p[..., :2])**2, axis = -1) < self.tol
                invalid = np.logical_or(np.any(np.isnan(cur_subset), axis = -1), ZNSSD[unsolved] > 1.) # mark the POI as invalid if the interpolated values contain np.nan or the ZNSSD is too large
            unsolved_index = np.where(unsolved)[0] # size num_unsolved array
            unsolved[unsolved_index[solved_dp]] = False
            unsolved[unsolved_index[invalid]] = False
            num_iter_list += unsolved.astype(floattype)
            num_unsolved = np.count_nonzero(unsolved)
            if num_unsolved == 0:
                break
        p = np.copy(p_ini)
        p[..., :2] = p_matrix[..., :2, 2]
        p[..., 2:] = (p_matrix[..., :2, :2] - np.eye(2, dtype = floattype)).reshape(-1, 4)
        p[np.isnan(ZNSSD)] = np.nan
        if self.print_info:
            t2 = time.time()
            print('{}/{} valid/all POIs: final/average interations = {:d}/{:.1f}'.format(np.count_nonzero(np.logical_not(np.isnan(ZNSSD))), ZNSSD.size, int(np.max(num_iter_list)), np.mean(num_iter_list)))
            if not np.all(np.isnan(ZNSSD)):
                print('ZNSSD: mean/max/std = {:.4f}/{:.4f}/{:.4f}'.format(np.nanmean(ZNSSD), np.nanmax(ZNSSD), np.nanstd(ZNSSD)))
            print('Subpixel searching time: {:.2f}s'.format(t2 - t1))
            print('******')
        return {'p' : p, 'ZNSSD' : ZNSSD,  'num_iteration': num_iter_list}

def gene_interpoints(points, p = None, subset_shift = None, subset_level = 10):
    """
    Generate interpolation points for POIs

    Parameters
    ----------
    points: ... x 2 array
        coordinates of POIs.

    subset_shift: None or 1 x subset_size x 2 array

    p: ... x 3 x 3 array or ... x 6 array
        p[i] is [[1 + ux, uy, u], [vx, 1+vy, v], [0, 0, 1]].

    subset_level: optional integer
        Used to generate subset_shift if it is None.

    Returns
    -------
    interpoints: ... x subset_size x 2 array.
        ... center POIs. Each POI corresponds subset_size points. Last dimension is x and y coordinates.
    """

    if subset_shift is None:
        subset_shift = gene_subset_shift(subset_level)

    if p is None:
        return np.expand_dims(points, -2) + subset_shift # ... x subset_size x 2 array
    elif p.shape[-1] == 3:
        return np.expand_dims(points, -2) + subset_shift + np.expand_dims(p[..., :2, 2], -2) + np.sum((np.expand_dims(p[..., :2, :2], -3) - np.eye(2, dtype = floattype))*np.expand_dims(subset_shift, -2), -1)
    elif p.shape[-1] == 6:
        return np.expand_dims(points, -2) + subset_shift + np.expand_dims(p[..., :2], -2) + np.sum(np.expand_dims(p[..., 2:].reshape(p.shape[:-1] + (2, 2)), -3)*np.expand_dims(subset_shift, -2), -1) # ... x subset_size x 2 array


def gene_subset_shift(subset_level, dtype = floattype):
    """
    Generate subset shift.

    Parameters
    ----------
    subset_level: int
        Half length of subset. Subset size = 2*subset_level + 1.

    Returns
    -------
    subset_shift: 1 x subset_size x 2 array
        Displacement shift for each point in a subset around the POI.
    """
    subset_shift = np.empty((2*subset_level + 1, 2*subset_level + 1, 2), dtype = dtype) # size (2*subset_level + 1) x (2*subset_level + 1) x 2 array. use to generate subset array for each POI
    subset_range = np.arange(-subset_level, subset_level + 1, dtype = dtype)
    subset_shift[:, :, 0] = np.reshape(subset_range, (1, -1))
    subset_shift[:, :, 1] = np.reshape(subset_range, (-1, 1))
    subset_shift = np.reshape(subset_shift, (1, -1, 2)) # 1 x subset_size x 2 array
    return subset_shift

def gene_random_subset_shift(subset_level, d = 0.2):
    subset_shift = gene_subset_shift(subset_level, dtype = floattype)
    return np.random.uniform(-d, d, size = subset_shift.shape).astype(floattype) + subset_shift

def bisection_search(reader, start_index, end_index, POIs, output_frame_list = None, subset_level = 10, ZNSSD_limit = 0.4, subpixel_method = 'fagn', lost_level = 0.25):
    # lost_level: if the frame step is larger than lost_level times the total frame range (end_index - start_index), then the determined lost POIs are not considered as lost. 
    lost_threshold = abs(end_index - start_index)*lost_level
    if start_index < end_index:
        direction = 'forward'
    elif start_index > end_index:
        direction = 'reverse'
    else:
        raise
    num_POI = len(POIs)
    bisection_task_graph = nx.DiGraph() # The directed graph to store the calculation process. Each edge represents one pixel and subpixel level calculation. Edge property 'POIs_list' contains the list of indexes of POIs that need to be calculated. This graph changes dynamically during the process following besection searching method. 
    bisection_task_graph.add_edge(start_index, end_index, POIs_list = np.arange(num_POI, dtype = np.int64)) # Add a calculation edge from start_index to end_index. The edge property 'POIs_list' contains the list of indexes of POIs need for this calculation, which is the full list of POIs in this case. 
    result_graph = nx.DiGraph() # The directed graph to store the calculation result. Node property 'POIs' contains the coordinates of available POIs at that frame index. The data contains np.nan that caused by either this POI does not need bisection searching at this frame index or the POI has lost. 
    result_graph.add_node(start_index, POIs = np.copy(POIs))
    POI_graph_list = [] # list of graphs that store the status of each POI. The node contains properties 'POI' (location of this POI at current frame index), 'A' (gradient matrix of this POI at current frame index), and 'status' ('fixed', 'calculated', or 'lost'). 
    A_identity = np.eye(2, dtype = floattype) # The identity gradient matrix. ux = uy = vx = vy = 0
    for i in range(num_POI):
        POI_graph = nx.DiGraph()
        POI_graph.add_node(start_index, POI = POIs[i], A = A_identity, status = 'fixed')
        POI_graph.graph['lost'] = False
        POI_graph.graph['POI_index'] = i
        POI_graph_list.append(POI_graph)
    
    # construct a subpixel level seaching object for reuse
    ref_img = reader[start_index]
    if subpixel_method == 'icgn':
        subpixel_search = Subpixel_ICGN(ref_img, ref_img, subset_level = subset_level)
    elif subpixel_method == 'fagn':
        subpixel_search = Subpixel_FAGN(ref_img, ref_img, subset_level = subset_level)
    else:
        raise
    subpixel_search.ref_index = start_index
    empty_POIs = np.empty_like(POIs) # location of POIs with np.nan
    empty_POIs[:] = np.nan
    
    while bisection_task_graph.size() > 0: # continue calculation if there is edge in bisection_task_graph
        # select the reference and current frame. The reference frame index is the first node with a child node, the current frame index is the child with smallest frame index. 
        if direction == 'forward':
            for frame_index in sorted(bisection_task_graph.nodes):
                if bisection_task_graph.out_degree(frame_index) > 0:
                    ref_index = frame_index
                    cur_index = min([v for u, v in bisection_task_graph.out_edges(frame_index)])
                    break
        else:
            for frame_index in sorted(bisection_task_graph.nodes)[::-1]:
                if bisection_task_graph.out_degree(frame_index) > 0:
                    ref_index = frame_index
                    cur_index = max([v for u, v in bisection_task_graph.out_edges(frame_index)])
                    break
        print('\nreference {}, current {}'.format(ref_index, cur_index))
        current_POIs_list = bisection_task_graph.edges[ref_index, cur_index]['POIs_list'] # get the list of POIs that need to calculate from the edge of bisection_task_graph
        current_POIs = result_graph.nodes[ref_index]['POIs'][current_POIs_list]
        if not cur_index in result_graph.nodes: # add empty POIs to cur_index in result_graph if it does not exist. 
            result_graph.add_node(cur_index, POIs = np.copy(empty_POIs))
        
        ref_img = reader[ref_index]
        cur_img = reader[cur_index]
        # pixel level searching
        liteflow = Pixel_liteflow(ref_img, cur_img, current_POIs)
        p_ini = liteflow.search()
        # subpixel level searching
        if subpixel_search.ref_index == ref_index: # if the ref_index has not been changed, then just use the previous fagn object and only update the POIs and cur_img. 
            subpixel_search.update_cur_info(cur_img)
        else:
            if subpixel_method == 'icgn':
                subpixel_search = Subpixel_ICGN(ref_img, cur_img, subset_level = subset_level)
            elif subpixel_method == 'fagn':
                subpixel_search = Subpixel_FAGN(ref_img, cur_img, subset_level = subset_level)
            subpixel_search.ref_index = ref_index
        bisection_task_graph.remove_edge(ref_index, cur_index) # remove caculated edge
        sub_res = subpixel_search.search(current_POIs, p_ini)
        p = sub_res['p'] # deformation parameters, shape ... x 6. [u, v, ux, uy, vx, vy]. May contain np.nan values, which represent invalid POI caused by moving out of frame. 
        ZNSSD = sub_res['ZNSSD'] # array of zero mean normalized sum of squared difference. If the value is smaller than the predefined threshold ZNSSD_limit, the the result is considered valid. May contain np.nan values, which represent invalid POI caused by moving out of frame. 
        
        # suppress RuntimeWarning of np.nan in the comparison
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid = np.logical_and(ZNSSD < ZNSSD_limit, np.logical_not(np.isnan(ZNSSD))) # location of valid result
            invalid = np.logical_not(valid)
        valid_POIs_list = current_POIs_list[valid] # list of valid POI index in this calculation step
        invalid_POIs_list = current_POIs_list[invalid] # list of valid POI index in this calculation step. Can either be not converged at current frame step with current initial guess, or moved out of frame. 
        valid_p = p[valid] # deformation parameters of valid POIs
        valid_ZNSSD = ZNSSD[valid]
        
        # store valid deformation information into POI_graph_list and result_graph
        result_graph.nodes[cur_index]['POIs'][valid_POIs_list] = result_graph.nodes[ref_index]['POIs'][valid_POIs_list] + valid_p[..., :2] # store localtions of calculated POIs in the cur_index into result_graph
        cur_POIs = result_graph.nodes[cur_index]['POIs']
        for i, POI_index in enumerate(valid_POIs_list):
            POI_p = valid_p[i]
            POI_graph = POI_graph_list[POI_index]
            POI_graph.add_edge(ref_index, cur_index, p = POI_p, ZNSSD = valid_ZNSSD[i]) # add edge and edge properties 'p' and 'ZNSSD'
            POI_graph.add_node(cur_index, POI = cur_POIs[POI_index], status = 'calculated', A = np.matmul(POI_p[2:].reshape(2, 2) + A_identity, POI_graph.nodes[ref_index]['A'])) # store the result in POI_graph. Change the status into 'calculated'. Gradient matrix is calculated by left multiple current result into previous gradient matrix. 
        
        if abs(cur_index - ref_index) > 1 and len(invalid_POIs_list) > 0:
        # if the frame index difference is larger than 1 and there are invalid POIs in the result
            invalid_ZNSSD = ZNSSD[invalid]
            if abs(cur_index - ref_index) > lost_threshold:
                lost_list = []
            else:
                lost_list =[POI_index for i, POI_index in enumerate(invalid_POIs_list) if np.isnan(invalid_ZNSSD[i])] # the POI is lost if the calculated deformation parameter contains np.nan. lost_list is the list of index of lost POI. 
            num_lost = len(lost_list)
            if num_lost > 0:
                # If there are lost POIs, mark the status of the POI as 'lost' in the POI_graph_list.
                for POI_index in lost_list:
                    POI_graph = POI_graph_list[POI_index]
                    POI_graph.add_node(cur_index, status = 'lost')
                    POI_graph.graph['lost'] = True
                    POI_graph.graph['lost_edge'] = ref_index, cur_index
    
                invalid_POIs_list = np.array([POI_index for POI_index in invalid_POIs_list if POI_index not in lost_list], dtype = np.int64) # Remove the lost POI from the invalid_POIs_list, so only unconverged POIs are left. 
                # remove the lost POIs in the POIs_list of remaining edges of  bisection_task_graph
                empty_edge_list = [] # list to store empty edges, and then remove them
                for u, v in bisection_task_graph.edges:
                    bisection_task_graph.edges[u, v]['POIs_list'] = np.array([POI_index for POI_index in bisection_task_graph.edges[u, v]['POIs_list'] if POI_index not in lost_list], dtype = np.int64)
                    if len(bisection_task_graph.edges[u, v]['POIs_list']) == 0:
                        empty_edge_list.append((u, v))
                if empty_edge_list:
                    bisection_task_graph.remove_edges_from(empty_edge_list)
    
            num_unsolved = len(invalid_POIs_list) # number of unsolved (e.g. unconverged) POIs
            if num_unsolved > 0 :
                middle_index = int(round((ref_index + cur_index)/2)) # calculate middle index for bisection seaching
                # add tow sub edges in the bisection_task_graph
                
                if (ref_index, middle_index) in bisection_task_graph.edges:
                    bisection_task_graph.edges[ref_index, middle_index]['POIs_list'] = np.unique(np.concatenate((bisection_task_graph.edges[ref_index, middle_index]['POIs_list'], invalid_POIs_list)))
                else:
                    bisection_task_graph.add_edge(ref_index, middle_index, POIs_list = invalid_POIs_list)
                if (middle_index, cur_index) in bisection_task_graph.edges:
                    bisection_task_graph.edges[middle_index, cur_index]['POIs_list'] = np.unique(np.concatenate((bisection_task_graph.edges[middle_index, cur_index]['POIs_list'], invalid_POIs_list)))
                else:
                    bisection_task_graph.add_edge(middle_index, cur_index, POIs_list = invalid_POIs_list)
        else:
        # if the frame index difference equals 1 or there is no invalid POIs, then no bisection searhing is required. 
            lost_list = invalid_POIs_list # all remaining POIs are marked as 'lost' because not further bisection searching is possible. 
            num_lost = len(lost_list)
            num_unsolved = 0
            if num_lost > 0:
                # mark the POI as 'lost'
                for POI_index in lost_list:
                    POI_graph = POI_graph_list[POI_index]
                    POI_graph.add_node(cur_index, status = 'lost')
                    POI_graph.graph['lost'] = True
                    POI_graph.graph['lost_edge'] = ref_index, cur_index
                
                empty_edge_list = []
                for u, v in bisection_task_graph.edges:
                    bisection_task_graph.edges[u, v]['POIs_list'] = np.array([POI_index for POI_index in bisection_task_graph.edges[u, v]['POIs_list'] if POI_index not in lost_list], dtype = np.int64) # remove lost POIs in POIs_list of every edge
                    if len(bisection_task_graph.edges[u, v]['POIs_list']) == 0:
                        empty_edge_list.append((u, v))
                if empty_edge_list:
                    bisection_task_graph.remove_edges_from(empty_edge_list)
    
        print('Bisection searching: solved {}, unsolved {}, lost {}'.format(len(valid_ZNSSD), num_unsolved, num_lost))
    
    if output_frame_list is not None:
        if direction == 'forward':
            output_frame_list = np.array(sorted([frame_index for frame_index in output_frame_list if frame_index > start_index and frame_index < end_index]), dtype = np.int64) # keep output frame index within start_index and end_index
        else:
            output_frame_list = np.array(sorted([frame_index for frame_index in output_frame_list if frame_index > end_index and frame_index < start_index]), dtype = np.int64) # keep output frame index within start_index and end_index
        # add empty POIs in output frame index of result_graph
        for frame_index in output_frame_list:
            if frame_index not in result_graph.nodes:
                result_graph.add_node(frame_index, POIs = np.copy(empty_POIs))
        
        # construct the output_task_graph to store the require calculation to get the locations of POIs at output frames. 
        output_task_graph = nx.DiGraph()
        output_task_graph.add_nodes_from(output_frame_list)
        output_task_graph.add_nodes_from([start_index, end_index])
        
        # determines the previous available node for each output frame index of each POI. The calculation path is kept at minimum to reduce the accumulated error. 
        for POI_index, POI_graph in enumerate(POI_graph_list):
            if direction == 'forward':
                if POI_graph.graph['lost']:
                    lost_index = POI_graph.graph['lost_edge'][1]
                    POI_frame_list = np.sort([frame_index for frame_index in POI_graph.nodes if POI_graph.nodes[frame_index]['status'] != 'lost']) # the list of frame indexes that location is available for this POI
                    current_output_frame_list = np.array([output_index for output_index in output_frame_list if output_index not in POI_frame_list and output_index < lost_index]) # current_output_frame_list does not contain calculated frame index and frame index larger than lost_index
                else:
                    POI_frame_list = np.sort(POI_graph.nodes)
                    current_output_frame_list = np.array([output_index for output_index in output_frame_list if output_index not in POI_frame_list])
                left_index_list = POI_frame_list[np.searchsorted(POI_frame_list, current_output_frame_list) - 1] # list of frame index on the left of each output index

                for ref_index, cur_index in zip(left_index_list, current_output_frame_list):
                    # Add the POI into the corresponding calculation edge. Creat the edge first if it does not exit. 
                    if (ref_index, cur_index) in output_task_graph.edges:
                        output_task_graph.edges[ref_index, cur_index]['POIs_list'].append(POI_index)
                    else:
                        output_task_graph.add_edge(ref_index, cur_index, POIs_list = [POI_index])
            else:
                if POI_graph.graph['lost']:
                    lost_index = POI_graph.graph['lost_edge'][1]
                    POI_frame_list = np.sort([frame_index for frame_index in POI_graph.nodes if POI_graph.nodes[frame_index]['status'] != 'lost']) # the list of frame indexes that location is available for this POI
                    current_output_frame_list = np.array([output_index for output_index in output_frame_list if output_index not in POI_frame_list and output_index > lost_index]) # current_output_frame_list does not contain calculated frame index and frame index smaller than lost_index
                else:
                    POI_frame_list = np.sort(POI_graph.nodes)
                    current_output_frame_list = np.array([output_index for output_index in output_frame_list if output_index not in POI_frame_list])
                right_index_list = POI_frame_list[np.searchsorted(POI_frame_list, current_output_frame_list)] # list of frame index on the left of each output index
                for ref_index, cur_index in zip(right_index_list, current_output_frame_list):
                    # Add the POI into the corresponding calculation edge. Creat the edge first if it does not exit. 
                    if (ref_index, cur_index) in output_task_graph.edges:
                        output_task_graph.edges[ref_index, cur_index]['POIs_list'].append(POI_index)
                    else:
                        output_task_graph.add_edge(ref_index, cur_index, POIs_list = [POI_index])
    
        # convert the POIs_list into numpy array for array indexing 
        for (ref_index, cur_index) in output_task_graph.edges:
            output_task_graph.edges[ref_index, cur_index]['POIs_list'] = np.array(output_task_graph.edges[ref_index, cur_index]['POIs_list'], dtype = np.int64)
        
        while output_task_graph.size() > 0: # continue calculation if there is edge in output_task_graph
            if direction == 'forward':
                for frame_index in sorted(output_task_graph.nodes):
                    if output_task_graph.out_degree(frame_index) > 0:
                        ref_index = frame_index
                        cur_index = min([v for u, v in output_task_graph.out_edges(frame_index)])
                        break
            else:
                for frame_index in sorted(output_task_graph.nodes)[::-1]:
                    if output_task_graph.out_degree(frame_index) > 0:
                        ref_index = frame_index
                        cur_index = max([v for u, v in output_task_graph.out_edges(frame_index)])
                        break
            print('\nreference {}, current {}'.format(ref_index, cur_index))
            current_POIs_list = output_task_graph.edges[ref_index, cur_index]['POIs_list']
            current_POIs = result_graph.nodes[ref_index]['POIs'][current_POIs_list]
            if cur_index not in result_graph.nodes:
                result_graph.add_node(cur_index, POIs = np.copy(empty_POIs))
            
            ref_img = reader[ref_index]
            cur_img = reader[cur_index]
            # pixel level searching
            liteflow = Pixel_liteflow(ref_img, cur_img, current_POIs)
            p_ini = liteflow.search()
            # subpixel level searching
            if subpixel_search.ref_index == ref_index: # if the ref_index has not been changed, then just use the previous fagn object and only update the POIs and cur_img. 
                subpixel_search.update_cur_info(cur_img)
            else:
                if subpixel_method == 'icgn':
                    subpixel_search = Subpixel_ICGN(ref_img, cur_img, subset_level = subset_level)
                elif subpixel_method == 'fagn':
                    subpixel_search = Subpixel_FAGN(ref_img, cur_img, subset_level = subset_level)
                subpixel_search.ref_index = ref_index
            output_task_graph.remove_edge(ref_index, cur_index) # remove caculated edge
            sub_res = subpixel_search.search(current_POIs, p_ini)
            p = sub_res['p'] # deformation parameters, shape ... x 6. [u, v, ux, uy, vx, vy]. May contain np.nan values, which represent invalid POI caused by moving out of frame. 
            ZNSSD = sub_res['ZNSSD'] # array of zero mean normalized sum of squared difference. If the value is smaller than the predefined threshold ZNSSD_limit, the the result is considered valid. May contain np.nan values, which represent invalid POI caused by moving out of frame. 
            with warnings.catch_warnings():
                # suppress RuntimeWarning of np.nan in the comparison
                warnings.simplefilter("ignore", category=RuntimeWarning)
                valid = np.logical_and(ZNSSD < ZNSSD_limit, np.logical_not(np.isnan(p[..., 0]))) # location of valid result
                invalid = np.logical_not(valid)
            valid_POIs_list = current_POIs_list[valid] # list of valid POI index in this calculation step
            valid_p = p[valid] # deformation parameters of valid POIs
            valid_ZNSSD = ZNSSD[valid]
            invalid_POIs_list = current_POIs_list[invalid] # list of valid POI index in this calculation step. Can either be not converged at current frame step with current initial guess, or moved out of frame. 
            
            # store valid deformation information into POI_graph_list
            result_graph.nodes[cur_index]['POIs'][valid_POIs_list] = result_graph.nodes[ref_index]['POIs'][valid_POIs_list] + valid_p[..., :2] # store localtions of calculated POIs in the cur_index into result_graph
            cur_POIs = result_graph.nodes[cur_index]['POIs']
            for i, POI_index in enumerate(valid_POIs_list):
                POI_p = valid_p[i]
                POI_graph = POI_graph_list[POI_index]
                POI_graph.add_edge(ref_index, cur_index, p = POI_p, ZNSSD = valid_ZNSSD[i]) # add edge and edge properties 'p' and 'ZNSSD'
                POI_graph.add_node(cur_index, POI = cur_POIs[POI_index], status = 'calculated', A = np.matmul(POI_p[2:].reshape(2, 2) + A_identity, POI_graph.nodes[ref_index]['A'])) # store the result in POI_graph. Change the status into 'calculated'. Gradient matrix is calculated by left multiple current result into previous gradient matrix. 
            
            if abs(cur_index - ref_index) > 1 and len(invalid_POIs_list) > 0:
            # if the frame index difference is larger than 1 and there are invalid POIs in the result
                invalid_ZNSSD = ZNSSD[invalid]
                if abs(cur_index - ref_index) > lost_threshold:
                    lost_list = []
                else:
                    lost_list =[POI_index for i, POI_index in enumerate(invalid_POIs_list) if np.isnan(invalid_ZNSSD[i])] # the POI is lost if the calculated deformation parameter contains np.nan. lost_list is the list of index of lost POI. 
                num_lost = len(lost_list)
                if num_lost > 0:
                    # If there are lost POIs, mark the status of the POI as 'lost' in the POI_graph_list.
                    for POI_index in lost_list:
                        POI_graph = POI_graph_list[POI_index]
                        POI_graph.add_node(cur_index, status = 'lost')
                        POI_graph.graph['lost'] = True
                        POI_graph.graph['lost_edge'] = ref_index, cur_index
        
                    invalid_POIs_list = np.array([POI_index for POI_index in invalid_POIs_list if POI_index not in lost_list], dtype = np.int64) # Remove the lost POI from the invalid_POIs_list, so only unconverged POIs are left. 
                    # remove the lost POIs in the POIs_list of remaining edges of  output_task_graph
                    empty_edge_list = []
                    for u, v in output_task_graph.edges:
                        output_task_graph.edges[u, v]['POIs_list'] = np.array([POI_index for POI_index in output_task_graph.edges[u, v]['POIs_list'] if POI_index not in lost_list], dtype = np.int64)
                        if len(output_task_graph.edges[u, v]['POIs_list']) == 0:
                            empty_edge_list.append((u, v))
                    if empty_edge_list:
                        output_task_graph.remove_edges_from(empty_edge_list)
                        
                num_unsolved = len(invalid_POIs_list) # number of unsolved (e.g. unconverged) POIs
                if num_unsolved > 0 :
                    middle_index = int(round((ref_index + cur_index)/2)) # calculate middle index for bisection seaching
                    # add tow sub edges in the bisection_task_graph
                    if (ref_index, middle_index) in output_task_graph.edges:
                        output_task_graph.edges[ref_index, middle_index]['POIs_list'] = np.unique(np.concatenate((output_task_graph.edges[ref_index, middle_index]['POIs_list'], invalid_POIs_list)))
                    else:
                        output_task_graph.add_edge(ref_index, middle_index, POIs_list = invalid_POIs_list)
                    if (middle_index, cur_index) in output_task_graph.edges:
                        output_task_graph.edges[middle_index, cur_index]['POIs_list'] = np.unique(np.concatenate((output_task_graph.edges[middle_index, cur_index]['POIs_list'], invalid_POIs_list)))
                    else:
                        output_task_graph.add_edge(middle_index, cur_index, POIs_list = invalid_POIs_list)
            else:
            # if the frame index difference equals 1 or there is no invalid POIs, then no bisection searhing is required. 
                lost_list = invalid_POIs_list # all remaining POIs are marked as 'lost' because not further bisection searching is possible. 
                num_lost = len(lost_list)
                num_unsolved = 0
                if num_lost > 0:
                    # mark the POI as 'lost'
                    for POI_index in lost_list:
                        POI_graph = POI_graph_list[POI_index]
                        POI_graph.add_node(cur_index, status = 'lost')
                        POI_graph.graph['lost'] = True
                        POI_graph.graph['lost_edge'] = ref_index, cur_index
                    
                    empty_edge_list = []
                    for u, v in output_task_graph.edges:
                        output_task_graph.edges[u, v]['POIs_list'] = np.array([POI_index for POI_index in output_task_graph.edges[u, v]['POIs_list'] if POI_index not in lost_list], dtype = np.int64) # remove lost POIs in POIs_list of every edge
                        if len(output_task_graph.edges[u, v]['POIs_list']) == 0:
                            empty_edge_list.append((u, v))
                    if empty_edge_list:
                        output_task_graph.remove_edges_from(empty_edge_list)
                
            print('Output searching: solved {}, unsolved {}, lost {}'.format(len(valid_ZNSSD), num_unsolved, num_lost))
    
    # strain calculation
    eye2 = np.eye(2, dtype = floattype)
    for frame_index in result_graph.nodes:
        green_strain = np.empty((num_POI, 4), floattype) # n x 4 array, [exx, exy, eyy, eeq]
        green_strain[:] = np.nan
        principal_stretch = np.empty((num_POI, 2), floattype)
        principal_stretch[:] = np.nan
        principal_vector = np.empty((num_POI, 2, 2), floattype)
        principal_vector[:] = np.nan
        true_strain = np.empty((num_POI, 2), floattype)
        true_strain[:] = np.nan
        for i, POI_graph in enumerate(POI_graph_list):
            if frame_index in POI_graph.nodes and 'A' in POI_graph.nodes[frame_index]:
                F = POI_graph.nodes[frame_index]['A'] # gradient matrix
                C = np.matmul(np.transpose(F), F) # right Cauchy-Green deformation tensor
                eig_vals, eig_vectors = np.linalg.eig(C)
                principal_stretch[i] = eig_vals**0.5
                principal_vector[i] = eig_vectors
                true_strain[i] = np.log(principal_stretch[i])
                green_matrix = (C - eye2)/2.
                green_strain[i, 0] = green_matrix[0, 0]
                green_strain[i, 1] = green_matrix[0, 1]
                green_strain[i, 2] = green_matrix[1, 1]
                green_strain[i, 3] = (green_strain[i, 0]**2 - green_strain[i, 0]*green_strain[i, 2] + green_strain[i, 2]**2 + 3*green_strain[i, 1]**2)**0.5
        result_graph.nodes[frame_index]['green_strain'] = green_strain
        result_graph.nodes[frame_index]['principal_stretch'] = principal_stretch
        result_graph.nodes[frame_index]['principal_vector'] = principal_vector
        result_graph.nodes[frame_index]['true_strain'] = true_strain
        
    return result_graph, POI_graph_list



def gene_POIs(exterior_vertex, step, list_interiors_vertex = None, round_int = True):
    '''
    exterior_vertex: n x 2 array. coordinate of vertexes of the exterior polygon.
    step: distance in px between neighbouring POIs.
    list_interiors_vertex: opional, None or list of n x 2 array
    
    return: {'POI':points, 'exterior_path':exterior_path, 'interiors_path':interiors_path}
    '''
    exterior_vertex = np.asarray(exterior_vertex, dtype = floattype)
    if np.any(exterior_vertex[0] != exterior_vertex[-1]):
        exterior_vertex = np.concatenate((exterior_vertex, np.expand_dims(exterior_vertex[0], axis = 0)))
    exterior_path = mpl.path.Path(exterior_vertex)
    bbox = np.round(exterior_path.get_extents().extents)
    xx, yy = np.meshgrid(np.linspace(bbox[0], bbox[2], num = round((bbox[2] - bbox[0])/step), dtype = floattype), np.linspace(bbox[1], bbox[3], num = round((bbox[3] - bbox[1])/step), dtype = floattype))

    xx = np.ravel(xx)
    yy = np.ravel(yy)
    points = np.stack((xx, yy), axis = 1).astype(floattype)
    points = points[exterior_path.contains_points(points)] # exclude points outside the exterior

    interiors_path = []
    if list_interiors_vertex:
        for i, interior_vertex in enumerate(list_interiors_vertex):
            interior_vertex = np.asarray(interior_vertex, dtype = floattype)
            if np.any(interior_vertex[0] != interior_vertex[-1]):
                interior_vertex = np.concatenate((interior_vertex, np.expand_dims(interior_vertex[0], axis = 0)))
                list_interiors_vertex[i] = interior_vertex
            interior_path = mpl.path.Path(interior_vertex, closed = True)
            interiors_path.append(interior_path)
            points = points[np.logical_not(interior_path.contains_points(points))] # exclude points inside interiors

    # add points on the edges
    for i, v1 in enumerate(exterior_vertex[:-1]):
        length = ((exterior_vertex[i+1, 0] - exterior_vertex[i, 0])**2 + (exterior_vertex[i+1, 1] - exterior_vertex[i, 1])**2)**0.5
        num = int(round(length/step))
        xx = np.linspace(exterior_vertex[i, 0], exterior_vertex[i+1, 0], num = num, endpoint=False, dtype = floattype)
        yy = np.linspace(exterior_vertex[i, 1], exterior_vertex[i+1, 1], num = num, endpoint=False, dtype = floattype)
        points = np.concatenate((points, np.stack((xx, yy), axis = 1)))
    if list_interiors_vertex:
        for interior_vertex in list_interiors_vertex:
            for i, v1 in enumerate(interior_vertex[:-1]):
                length = ((interior_vertex[i+1, 0] - interior_vertex[i, 0])**2 + (interior_vertex[i+1, 1] - interior_vertex[i, 1])**2)**0.5
                num = int(round(length/step))
                xx = np.linspace(interior_vertex[i, 0], interior_vertex[i+1, 0], num = num, endpoint=False, dtype = floattype)
                yy = np.linspace(interior_vertex[i, 1], interior_vertex[i+1, 1], num = num, endpoint=False, dtype = floattype)
                points = np.concatenate((points, np.stack((xx, yy), axis = 1)))

    # remove points in a pair that distance is less than step*0.5
    tree = spatial.ckdtree.cKDTree(points)
    remove_pair = np.array(list(tree.query_pairs(step*0.5)), dtype = np.int64)

    if len(remove_pair) > 0 :
        points = np.delete(points, remove_pair[:, 0], axis = 0)
    if round_int == True:
        points = points.round()
    return points

def refine_POIs(POIs, ref_img, subset_level, refine_level = 5, remain_percentile = None):
    """
    Move the locations of POIs around the original locatrions to increase the contrast of subsets.

    Parameters
    ----------
    POIs: n x 2 array
        Coordinates of POIs.

    ref_img: row x col array
        reference image

    subset_level: int
        subset level to calculate SSSIG

    refine_level: int
        Maximum distance to move POIs around

    remain_percentile: float or None
        Remove SSSIG with lower remain_percentile%. None is do not removing.

    Return
    ----------
    keypoints: n_feature x 2 array
        Coordinates of detected features

    descriptors: n_feature x 8 uint32 array
    """

    # refine the locations of points of interest to increase the gradient inside the subsets.
    # remain_percentile:
    POIs = np.round(POIs).astype(np.int64)
    temp_icgn = Subpixel_ICGN(ref_img, ref_img, subset_level = 11, revision = 'adjust')
    fx_window = util.shape.view_as_windows(temp_icgn.fx, 2*subset_level + 1)
    fy_window = util.shape.view_as_windows(temp_icgn.fy, 2*subset_level + 1)

    shift_y, shift_x = np.mgrid[-refine_level:refine_level + 1, -refine_level:refine_level + 1]
    shift = np.expand_dims(np.stack((np.ravel(shift_y), np.ravel(shift_x)), axis = 1).astype(np.int64), 1)
    POIs_candidate = POIs + shift # m x n x 2

    SSSIG_candidate = np.sum((fx_window[POIs_candidate[..., 1] - subset_level, POIs_candidate[..., 0] - subset_level])**2 + (fy_window[POIs_candidate[..., 1] - subset_level, POIs_candidate[..., 0] - subset_level])**2, axis = (-1, -2))
    larger_SSSIGG_index = np.argmax(SSSIG_candidate, axis = 0)
    del temp_icgn
    larger_SSSIGG = SSSIG_candidate[larger_SSSIGG_index, np.arange(len(POIs))]
    if remain_percentile is not None:
        remain_SSSIGG = larger_SSSIGG > np.percentile(larger_SSSIGG, [remain_percentile])[0]
        return POIs_candidate[larger_SSSIGG_index[remain_SSSIGG], np.arange(len(POIs))[remain_SSSIGG]].astype(floattype)
    else:
        return POIs_candidate[larger_SSSIGG_index, np.arange(len(POIs))].astype(floattype)

converter_func = lambda s : float(s[1:-1])
class Tensile_data_loader(object):
    """
    Load tensile data generated from the csv file of the instron machine. The format of the file is (with default settings)
    Time	Extension	Load	Tensile strain (Extension)	Tensile stress	Tensile extension
    (s)	(mm)	(N)	(mm/mm)	(MPa)	(mm)


    """
    
    def __init__(self, tensile_data_file):
        self.raw_data = np.genfromtxt(tensile_data_file, delimiter=",", skip_header=5, converters = {0:converter_func, 1:converter_func, 2:converter_func, 3:converter_func, 4:converter_func, 5:converter_func}, names = ['time','extension','load','tensile_strain','tensile_stress','tensile_extension'])

        max_load_index = np.argmax(self.raw_data['load'])
        self.cross_area = self.raw_data['load'][max_load_index]/self.raw_data['tensile_stress'][max_load_index] # area of the cross section in mm^2
        self.time = self.raw_data['time']
        self.stress = self.raw_data['tensile_stress']
        self.tensile_strain = self.raw_data['tensile_strain']

        self.time_shift = 0.
        self.inter_func = interpolate.interp1d(self.time, self.stress, bounds_error = False)
        self.inter_strain = interpolate.interp1d(self.time, self.tensile_strain, bounds_error = False)

    def inter_stress(self, time_array):
        return self.inter_func(time_array)

    def set_time_shift(self, time_shift):
        self.time_shift = time_shift

    def inter_stress_corrected(self, time_array):
        return self.inter_func(np.asarray(time_array) - self.time_shift)

    def inter_strain(self, time_array):
        return self.inter_strain(time_array)

    def inter_strain_corrected(self, time_array):
        return self.inter_strain(np.asarray(time_array) - self.time_shift)

class video_loader(object):
    def __init__(self, video_file, fps = 30000/1001, maximum_frame = None):
        self.reader = imageio.get_reader(video_file, mode = 'I')
        if maximum_frame:
            self.nframes = maximum_frame
        else:
            self.nframes = self.reader.get_length() - 5
        self.relative_timestamp_list = np.arange(self.nframes, dtype = floattype)/fps

    def __getitem__(self, index):
        return self.reader.get_data(index)

class image_folder_loader(object):
    def __init__(self, image_folder, suffix, time_format = None, rescale = None):
        self.suffix = str(suffix).lower()
        self.image_folder = image_folder
        self.img_files = [image for image in natsort.natsorted(os.listdir(self.image_folder)) if os.path.isfile(os.path.join(self.image_folder, image)) and image.endswith(suffix)]
        self.num_frame = len(self.img_files)
        if self.num_frame == 0:
            raise
        if time_format:
            self.time_list = [datetime.datetime.strptime('.'.join(image.split('.')[:-1]), time_format) for image in self.img_files]
            self.timestamp_list = [img_tiem.timestamp() for img_tiem in self.time_list]
            self.timestamp_list = np.array(self.timestamp_list)
            self.relative_timestamp_list = self.timestamp_list - self.timestamp_list[0]
        self.rescale = rescale
    
    def __len__(self):
        return self.num_frame

    def __getitem__(self, index):
        filename = os.path.join(self.image_folder, self.img_files[index])
        if self.suffix == 'npy':
            if self.rescale:
                return (np.load(filename) + 1.)/self.rescale + 1.
            else:
                return np.load(filename)
        elif self.suffix == 'jpg' or self.suffix == 'tiff' or self.suffix == 'tif':
            return imageio.imread(filename)
        elif self.suffix == 'save' or self.suffix == 'dic':
            return pickle.load(open(filename, 'rb'))

def plot_img_points(img, points = None, size = 1, bound_length = None, boundbox = None, maximize_window = True):
    # bound_length: length add to the bound box of points
    # size: default size of points is 1/100 of the minimum length of the image dimension. Define size to change it relatively.
    row, col = img.shape[:2]
    if boundbox is not None:
        xmin, xmax, ymin, ymax = boundbox
    elif bound_length is not None:
        xmin, ymin = np.min(points, axis = 0) - [bound_length, bound_length]
        xmax, ymax = np.max(points, axis = 0) + [bound_length, bound_length]
        xmin = int(xmin) if xmin > 0 else 0
        ymin = int(ymin) if ymin > 0 else 0
        xmax = int(xmax) if xmax < col else col
        ymax = int(ymax) if ymax < row else row
    else:
        xmin = 0
        xmax = col
        ymin = 0
        ymax = row

    if img.ndim == 3:
        plt.imshow(img[ymin:ymax, xmin:xmax])
    else:
        plt.imshow(img[ymin:ymax, xmin:xmax], cmap = 'gray')
    if points is not None:
        points  = points - (xmin, ymin)
        patches = [mpl.patches.Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(points[:, 0], points[:, 1], size*min(xmax - xmin, ymax - ymin)/200)]
        collection = mpl.collections.PatchCollection(patches)
        plt.gca().add_collection(collection)

    if maximize_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

def fig2img(fig, dpi = 300):
    # convert matplotlib fig into numpy array
    fig.dpi = dpi
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width); height = int(height)
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image

def on_click(event):
    # get the x and y coords, flip y from top to bottom
    x, y = event.xdata, event.ydata
    ax = plt.gca()
    cir = mpl.patches.Circle(xy=(x, y), radius = 4, color = 'red')
    ax.add_artist(cir)
    print('[{}, {}],'.format(x, y))

def plot_img_click(img, points = None, size = 0.6):
    plot_img_points(img, points = points, size = size)
    plt.connect('button_press_event', on_click)

def plot_img_points_trip(img, points, variable, bound_length = None, boundbox = None, vmin = None, vmax = None, plot_points = False, alpha = 0.6, colorbar_orientation = 'horizontal'):
    # identify locations with NaN and then remove them
    nonnan_location = np.logical_not(np.isnan(variable))
    points = points[nonnan_location]
    variable = variable[nonnan_location]

    img = np.asarray(img)
    row, col = img.shape[:2]
    if boundbox is not None:
        xmin, xmax, ymin, ymax = boundbox
    elif bound_length is not None:
        xmin, ymin = np.min(points, axis = 0) - [bound_length, bound_length]
        xmax, ymax = np.max(points, axis = 0) + [bound_length, bound_length]
        xmin = int(xmin) if xmin > 0 else 0
        ymin = int(ymin) if ymin > 0 else 0
        xmax = int(xmax) if xmax < col else col
        ymax = int(ymax) if ymax < row else row
        boundbox = xmin, xmax, ymin, ymax
    else:
        xmin = 0
        xmax = col
        ymin = 0
        ymax = row
        boundbox = xmin, xmax, ymin, ymax
    plt.tripcolor(points[:, 0] - xmin, points[:, 1] - ymin, variable, alpha = alpha, vmin = vmin, vmax = vmax, shading = 'gouraud', cmap = 'jet')
#    plt.tripcolor(points[:, 0] - xmin, points[:, 1] - ymin, variable, alpha = alpha, vmin = vmin, vmax = vmax, shading = 'gouraud')
    if colorbar_orientation == 'horizontal':
#        plt.colorbar(orientation='horizontal', format='%.2e')
        plt.colorbar(orientation='horizontal')
    elif colorbar_orientation == 'virtical':
        plt.colorbar(orientation='virtical')

    if plot_points == True:
        plot_img_points(img, points = points, bound_length = bound_length, boundbox = boundbox)
    else:
        plot_img_points(img, points = None, bound_length = bound_length, boundbox = boundbox)

def gene_movie(reader, output_res, video_filename, video_time, video_fps, plot_command, output_frame_list, dpi = 300):
    n_video_frame = video_time*video_fps
    n_output_frame = len(output_frame_list)
    if n_video_frame >= n_output_frame:
        print('do not drop frame')
        video_fps = int(round(n_output_frame/video_time))
        video_frame_list = output_frame_list
    else:
        print('drop frame')
        video_frame_list = []
        for frame_index in np.linspace(output_frame_list[0], output_frame_list[-1], n_video_frame, dtype = np.int64):
            right_index = np.searchsorted(output_frame_list, frame_index)
            if frame_index in output_frame_list:
                video_frame_list.append(frame_index)
            elif frame_index - output_frame_list[right_index - 1] > output_frame_list[right_index] - frame_index:
                video_frame_list.append(output_frame_list[right_index])
            else:
                video_frame_list.append(output_frame_list[right_index - 1])

    plt.ioff()
    with imageio.get_writer(video_filename, mode='I', format='FFMPEG', fps = video_fps, quality = 10) as writer:
        for index in video_frame_list:
            print(index)
            fig = plt.figure()
            exec(plot_command)
            plt.close()
            writer.append_data(fig2img(fig, dpi))

from scipy import fftpack
def img_shift_pre(shape):
    """
    Calculate (Nc, Nr) to prepare for img_shift. 

    Parameters
    ----------
    shape : row, col
        Shape of the image
    """
    Nr_fbase = fftpack.ifftshift(np.arange(-math.floor(shape[0]/2), math.ceil(shape[0]/2), dtype = floattype))
    Nc_fbase = fftpack.ifftshift(np.arange(-math.floor(shape[1]/2), math.ceil(shape[1]/2), dtype = floattype))
    return np.meshgrid(Nc_fbase, Nr_fbase)

def img_shift(img, shift_x, shift_y, NcNr = None):
    '''
    Generate shifted image by FFT
    '''
    row, col = img.shape[:2]
    if NcNr is None:
        Nc, Nr = img_shift_pre((row, col))
    else:
        Nc, Nr = NcNr
    img_spe = fftpack.fft2(img)
    return np.real(fftpack.ifft2(img_spe*np.exp(1j*2*math.pi*(-shift_y*Nr/row - shift_x*Nc/col))))
