"""
Find nearest neighbor and got crop patches based on registration of nearest neighbor 


Sample usage:

import torch
from utils import *
from torch.utils.data import Dataset, DataLoader
from utils.registration import *

def multi_masks2onehot(mask):
    D, H, W = mask.shape
    onehot_mask = np.zeros((2, D, H, W))
    for i in range(2):
        onehot_mask[i][mask == i] = 1

    return onehot_mask

class ValBrainImageDataset(Dataset):

    def __init__(self, ls, moving_ls, transform=None):

        self.ls = ls
        self.moving_ls = moving_ls
        self.transform = transform

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, idx):   
        print self.ls[idx]
#         start_time = time.time()
        roi, img = get_patient_img_and_contour(self.ls[idx])
        nearest_p = find_nearest_patient(self.ls[idx], self.moving_ls)
        fix_contour, fix_img = get_patient_img_and_contour(nearest_p)
        outTx = affine(img, fix_img)
        out = resample(img, fix_img, outTx)
        out_contour = resample(roi, fix_contour, outTx, interpolator = sitk.sitkNearestNeighbor)
        indice = find_border_dynamic_threshold(sitk.GetArrayFromImage(out_contour))
        indices = pad3D_indices(indice, sitk.GetArrayFromImage(out_contour))
    
        cropped_img =  img[indices[0][0]:indices[0][-1],indices[1][0]:indices[1][-1],indices[2][0]:indices[2][-1]].astype(np.float32)
        cropped_roi = multi_masks2onehot(roi[indices[0][0]:indices[0][-1],indices[1][0]:indices[1][-1],indices[2][0]:indices[2][-1]]).astype(np.float32)
        # print("--- %s seconds ---" % (time.time() - start_time))

        return torch.from_numpy(cropped_img[np.newaxis, ...]).type(torch.FloatTensor), torch.from_numpy(cropped_roi).type(torch.FloatTensor) 
"""

import SimpleITK as sitk
import os
import numpy as np


def find_nearest_patient(p_name, p_ls):
    """
    First to loop through the training name list to get the nearest neighbor based on pixel intensity histogram and z-slices num
    Args:
        p_name: patient name(str)
        p_ls: list of patient names(str) to get nearest neighbor
    Returns:
        patient name of nearest neighbor(str)
    """
    contour, img = get_patient_img_and_contour(p_name)
    distance_dict = {}
    for i in range(len(p_ls)):
        if p_ls[i] != p_name:
#             print p_ls[i]
            contour2, img2 = get_patient_img_and_contour(p_ls[i])
            if np.abs(img.shape[0] - img2.shape[0]) <= 11: 
                hist1 = np.histogram(img,bins=1000)[0]
                hist2 = np.histogram(img2,bins=1000)[0]
                distance_dict[p_ls[i]] = (calculateDistance(hist1, hist2))
    return min(distance_dict, key = lambda x: distance_dict.get(x))

def get_patient_img_and_contour(patient_name, contour = 'parotid r', img_path = '/mnt/hdd10T/htang6/data/brain_ai/preprocessed/'):
    """
    Read patient image and contour in np.array
    Args:
        patient_name: patient name(str)
        contour: name of contour(str)
        img_path: path to processed image data and contour data
    Returns:
        contour and ct image in np.array
    """
    for f in os.listdir(img_path):
        if f.startswith(patient_name) and contour in f.lower():
            interestarea = np.load(img_path + f)
        elif f.startswith(patient_name) and 'clean' in f.lower():
            image = np.load(img_path + f)

    return interestarea, image

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)

def  find_border_dynamic_threshold(mask):
    """
    Find the border of the passed in mask
    Args:
        mask: mask in np.array
    Returns:
        indices of elements value are bigger than zero in np.array
    """
    threshold = 0
    indices = np.where(mask > threshold)
    return indices

def pad3D_indices(indices, original_img, shape= (30, 120, 120)):
    """
    Pass in the indices of border, pad indices into desired shape
    Args:
        indices: indices from find_border_dynamic_threshold function.
        original_img: original_img in np.array
        shape: desired shape of crop patches
    Returns:
        padded indices fit the desired shape(list of tuples)
    """
    m_max = indices[0].max()
    m_min = indices[0].min()
    n_max = indices[1].max()
    n_min = indices[1].min()
    r_max = indices[2].max()
    r_min = indices[2].min()
#     print a[0]
    m_diff = shape[0] - (m_max - m_min)
    n_diff = shape[1] - (n_max - n_min)
    r_diff = shape[2] - (r_max - r_min)
    z_min = m_min - m_diff/2
    z_max = m_max + (m_diff+1)/2
    y_min = n_min - n_diff/2 #- 10
    y_max = n_max + (n_diff+1)/2#- 10
    x_min = r_min - r_diff/2 
    x_max = r_max + (r_diff+1)/2
    if z_max > original_img.shape[0]:
        z_max = original_img.shape[0]
        z_min = z_max - shape[0]
    if y_max > original_img.shape[1]:
        y_max = original_img.shape[1]
        y_min = y_max - shape[1]
    if x_max > original_img.shape[2]:
        x_max = original_img.shape[2]
        x_min = x_max - shape[2]
    if z_min < 0:
        z_min = 0
        z_max = shape[0]
    if y_min < 0:
        y_min = 0
        y_max = shape[1]
    if x_min < 0:
        x_min = 0
        x_max = shape[2]

    return [(z_min, z_max) ,(y_min, y_max), (x_min, x_max)]

def rigid(fixed, moving):
    """
    rigid registration on the crop patch
    Args:
        fixed: fixed image in np.array.
        moving: moving image in np.array.
    Returns:
        displacement field
    """
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(8.0, .01, 200 )
    R.SetInitialTransform(sitk.TranslationTransform(sitk.GetImageFromArray(fixed).GetDimension()))
    R.SetInterpolator(sitk.sitkNearestNeighbor)

    R.AddCommand( sitk.sitkIterationEvent, lambda: R )

    outTx = R.Execute(sitk.GetImageFromArray(fixed), sitk.GetImageFromArray(moving))
    
    return outTx

def affine(fixed, moving):
    """
    affine registration on the crop patch
    Args:
        fixed: fixed image in np.array.
        moving: moving image in np.array.
    Returns:
        displacement field
    """
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
#     initial_transform = sitk.CenteredTransformInitializer(sitk.GetImageFromArray(fixed), 
#                                                       sitk.GetImageFromArray(moving), 
#                                                       sitk.AffineTransform(sitk.GetImageFromArray(fixed).GetDimension()))
#     R.SetShrinkFactorsPerLevel([3,2,1])
#     R.SetSmoothingSigmasPerLevel([2,1,1])

#     R.SetMetricAsJointHistogramMutualInformation(20)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetOptimizerAsGradientDescent(learningRate=0.5,
                                    numberOfIterations=200,
                                    estimateLearningRate = R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(sitk.AffineTransform(sitk.GetImageFromArray(fixed).GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)
#     R.SetInitialTransform(sitk.AffineTransform(sitk.GetImageFromArray(fixed).GetDimension()))
#     R.SetInterpolator(sitk.sitkNearestNeighbor)

    R.AddCommand( sitk.sitkIterationEvent, lambda:R )

    outTx = R.Execute(sitk.GetImageFromArray(fixed), sitk.GetImageFromArray(moving))
    
    return outTx

def resample(fixed, moving, outTx, interpolator = sitk.sitkLinear):
    """
    Apply the displacement field onto moving image 
    Args:
        fixed: fixed image in np.array.
        moving: moving image in np.array.
        outTx: displacement field.
        interpolator: sitk.sitkLinear by defalut
    Returns:
        moving image after registration in sitk.Image
    """
#   outTx = sitk.ReadTransform('out.txt')
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(fixed));
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(sitk.GetImageFromArray(moving))
    
    return out
