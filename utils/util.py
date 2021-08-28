import sys
import numpy as np
import torch
import pydicom as dicom
import numpy as np
from scipy.sparse import csc_matrix
from collections import defaultdict
import os
import shutil
import operator
import warnings
import numpy as np
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
import matplotlib.cm as cm
import math
from skimage import measure
import scipy
from scipy.ndimage import zoom
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import pandas as pd
import yaml
import cv2
import copy
try:
    # Python2
    from StringIO import StringIO
except ImportError:
    # Python3
    from io import StringIO


def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    order: int, order for resample function scipy.ndimage.interpolation.zoom
    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode='nearest', order=order)

    return (image_new, resample_spacing)


class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def load_yaml(path):
    class Struct:
        def __init__(self, **entries): 
            self.__dict__.update(entries)

    with open(path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    data_class = Struct(**data_dict)

    return data_dict, data_class


def py_nms(dets, thresh):
    # Check the input dtype
    if isinstance(dets, torch.Tensor):
        if dets.is_cuda:
            dets = dets.cpu()
        dets = dets.data.numpy()
        
    z = dets[:, 1]
    y = dets[:, 2]
    x = dets[:, 3]
    d = dets[:, 4]
    h = dets[:, 5]
    w = dets[:, 6]
    scores = dets[:, 0]

    areas = d * h * w
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx0 = np.maximum(x[i] - w[i] / 2., x[order[1:]] - w[order[1:]] / 2.)
        yy0 = np.maximum(y[i] - h[i] / 2., y[order[1:]] - h[order[1:]] / 2.)
        zz0 = np.maximum(z[i] - d[i] / 2., z[order[1:]] - d[order[1:]] / 2.)
        xx1 = np.minimum(x[i] + w[i] / 2., x[order[1:]] + w[order[1:]] / 2.)
        yy1 = np.minimum(y[i] + h[i] / 2., y[order[1:]] + h[order[1:]] / 2.)
        zz1 = np.minimum(z[i] + d[i] / 2., z[order[1:]] + d[order[1:]] / 2.)

        inter_w = np.maximum(0.0, xx1 - xx0)
        inter_h = np.maximum(0.0, yy1 - yy0)
        inter_d = np.maximum(0.0, zz1 - zz0)
        intersect = inter_w * inter_h * inter_d
        overlap = intersect / (areas[i] + areas[order[1:]] - intersect)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return torch.from_numpy(dets[keep]), torch.LongTensor(keep)


def py_box_overlap(boxes1, boxes2):
    overlap = np.zeros((len(boxes1), len(boxes2)))

    z1, y1, x1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2]
    d1, h1, w1 = boxes1[:, 3], boxes1[:, 4], boxes1[:, 5]
    areas1 = d1 * h1 * w1

    z2, y2, x2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2]
    d2, h2, w2 = boxes2[:, 3], boxes2[:, 4], boxes2[:, 5]
    areas2 = d2 * h2 * w2

    for i in range(len(boxes1)):
        xx0 = np.maximum(x1[i] - w1[i] / 2., x2 - w2 / 2.)
        yy0 = np.maximum(y1[i] - h1[i] / 2., y2 - h2 / 2.)
        zz0 = np.maximum(z1[i] - d1[i] / 2., z2 - d2 / 2.)
        xx1 = np.minimum(x1[i] + w1[i] / 2., x2 + w2 / 2.)
        yy1 = np.minimum(y1[i] + h1[i] / 2., y2 + h2 / 2.)
        zz1 = np.minimum(z1[i] + d1[i] / 2., z2 + d2 / 2.)

        inter_w = np.maximum(0.0, xx1 - xx0)
        inter_h = np.maximum(0.0, yy1 - yy0)
        inter_d = np.maximum(0.0, zz1 - zz0)
        intersect = inter_w * inter_h * inter_d
        overlap[i] = intersect / (areas1[i] + areas2 - intersect)

    return overlap


def center_box_to_coord_box(bboxes):
    """
    Convert bounding box using center of rectangle and side lengths representation to 
    bounding box using coordinate representation
    [center_z, center_y, center_x, D, H, W] -> [z_start, y_start, x_start, z_end, y_end, x_end]

    bboxes: list of bounding boxes, [num_bbox, 6]
    """
    res = np.zeros(bboxes.shape)
    res[:, 0] = bboxes[:, 0] - bboxes[:, 3] / 2.
    res[:, 1] = bboxes[:, 1] - bboxes[:, 4] / 2.
    res[:, 2] = bboxes[:, 2] - bboxes[:, 5] / 2.
    res[:, 3] = bboxes[:, 0] + bboxes[:, 3] / 2.
    res[:, 4] = bboxes[:, 1] + bboxes[:, 4] / 2.
    res[:, 5] = bboxes[:, 2] + bboxes[:, 5] / 2.

    return res


def coord_box_to_center_box(bboxes):
    """
    Convert bounding box using coordinate representation to 
    bounding box using center of rectangle and side lengths representation
    [z_start, y_start, x_start, z_end, y_end, x_end] -> [center_z, center_y, center_x, D, H, W]

    bboxes: list of bounding boxes, [num_bbox, 6]
    """
    res = np.zeros(bboxes.shape)

    res[:, 3] = bboxes[:, 3] - bboxes[:, 0]
    res[:, 4] = bboxes[:, 4] - bboxes[:, 1]
    res[:, 5] = bboxes[:, 5] - bboxes[:, 2]
    res[:, 0] = bboxes[:, 0] + res[:, 3] / 2.
    res[:, 1] = bboxes[:, 1] + res[:, 4] / 2.
    res[:, 2] = bboxes[:, 2] + res[:, 5] / 2.

    return res

def ext2factor(bboxes, factor=8):
    """
    Given center box representation which is [z_start, y_start, x_start, z_end, y_end, x_end],
    return closest point which can be divided by 8 
    """
    bboxes[:, :3] = bboxes[:, :3] // factor * factor
    bboxes[:, 3:] = bboxes[:, 3:] // factor * factor + (bboxes[:, 3:] % factor != 0).astype(np.int32) * factor

    return bboxes

def clip_boxes(boxes, img_size):
    '''
    clip boxes outside the image, all box follows [z_start, y_start, x_start, z_end, y_end, x_end]
    '''
    depth, height, width = img_size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, depth)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, depth)
    boxes[:, 4] = np.clip(boxes[:, 4], 0, height)
    boxes[:, 5] = np.clip(boxes[:, 5], 0, width)

    return boxes


def detections2mask(detections, masks, img_reso, num_class=28):
    """
    Apply results of mask-rcnn (detections and masks) to mask result.

    detections: detected bounding boxes [z, y, x, d, h, w, category]
    masks: mask predictions correponding to each one of the detections config['mask_crop_size']
    img_reso: tuple with 3 elements, shape of the image or target resolution of the mask
    """
    D, H, W = img_reso
    mask = np.zeros((num_class, D, H, W))
    for i in range(len(detections)):
        z, y, x, d, h, w, cat = detections[i]

        cat = int(cat)
        z_start = max(0, int(np.floor(z - d / 2.)))
        y_start = max(0, int(np.floor(y - h / 2.)))
        x_start = max(0, int(np.floor(x - w / 2.)))
        z_end = min(D, int(np.ceil(z + d / 2.)))
        y_end = min(H, int(np.ceil(y + h / 2.)))
        x_end = min(W, int(np.ceil(x + w / 2.)))

        m = masks[i]
        D_c, H_c, W_c = m.shape
        zoomed_crop = zoom(m, 
                    (float(z_end - z_start) / D_c, float(y_end - y_start) / H_c, float(x_end - x_start) / W_c), 
                    order=2)
        mask[cat - 1][z_start:z_end, y_start:y_end, x_start:x_end] = (zoomed_crop > 0.5).astype(np.uint8)
    
    return mask


def crop_boxes2mask(crop_boxes, masks, img_reso, num_class=28):
    """
    Apply results of mask-rcnn (detections and masks) to mask result.

    crop_boxes: detected bounding boxes [z, y, x, d, h, w, category]
    masks: mask predictions correponding to each one of the detections config['mask_crop_size']
    img_reso: tuple with 3 elements, shape of the image or target resolution of the mask
    """
    D, H, W = img_reso
    mask = np.zeros((num_class, D, H, W))
    for i in range(len(crop_boxes)):
        z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]

        cat = int(cat)

        m = masks[i]
        D_c, H_c, W_c = m.shape
        mask[cat - 1][z_start:z_end, y_start:y_end, x_start:x_end] = (m > 0.5).astype(np.uint8)
    
    return mask



def annotation2masks(mask, roi_names=None):
    D, H, W = mask[list(mask.keys())[0]].shape
    masks = np.zeros([len(roi_names), D, H, W])
    for i, roi in enumerate(roi_names):
        if roi in mask:
            masks[i][mask[roi] > 0] = 1


def masks2bboxes_masks(masks, border):
    """
    Generate bounding boxes from masks

    masks: [num_class, D, H, W]
    return: [z, y, x, class]
    """
    num_class, D, H, W = masks.shape
    bboxes = []
    truth_masks = []
    for i in range(num_class):
        mask = masks[i]
        if np.any(mask):
            zz, yy, xx = np.where(mask)
            bboxes.append([(zz.max() + zz.min()) / 2., (yy.max() + yy.min()) / 2., (xx.max() + xx.min()) / 2., 
                zz.max() - zz.min() + 1 + border / 2, yy.max() - yy.min() + 1 + border, xx.max() - xx.min() + 1 + border, i + 1])
            truth_masks.append(mask)

    return bboxes, truth_masks


def get_contours_from_masks(masks):
    """
    Generate contours from masks by going through each organ slice by slice
    
    masks: [num_class, D, H, W]
    return: contours of shape [num_class, D, H, W] for each organ
    """
    contours = np.zeros(masks.shape, dtype=np.uint8)
    
    # Iterate all organs/channels
    for i, mask in enumerate(masks):
        # For each organ, Iterate all slices
        for j, s in enumerate(mask):
            c = np.zeros(s.shape)
            pts = measure.find_contours(s, 0)

            if pts:
                # There is contour in the image
                pts = np.concatenate(pts).astype(np.int32)
                for point in pts:
                    c[point[0], point[1]] = 1

            contours[i][j] = c
            
    return contours


def merge_contours(contours):
    """
    Merge contours for each organ into one ndimage, overlapped pixels will
    be override by the later class value
    
    contours: [num_class, D, H, W]
    return: merged contour of shape [D, H, W]
    """
    num_class, D, H, W = contours.shape
    merged_contours = np.zeros((D, H, W), dtype=np.uint8)
    for i in range(num_class):
        merged_contours[contours[i] > 0] = i + 1
    
    return merged_contours


def merge_masks(masks):
    """
    Merge masks for each organ into one ndimage, overlapped pixels will
    be override by the later class value
    
    contours: [num_class, D, H, W]
    return: merged contour of shape [D, H, W]
    """
    num_class, D, H, W = masks.shape
    merged_masks = np.zeros((D, H, W), dtype=np.uint8)
    for i in range(num_class):
        merged_masks[masks[i] > 0] = i + 1
    
    return merged_masks


def dice_score(y_pred, y_true, num_class=1, decimal=4):
    res = []
    for i in range(num_class):
        target = y_true == i
        pred = y_pred == i
        if target.sum():
            score = 2 * (target * pred).sum() / float((target.sum() + pred.sum()))
            res.append(round(score, decimal))
        else:
            res.append(None)

    return res


def dice_score_seperate(y_pred, y_true, num_class=1, decimal=4):
    res = []
    for i in range(num_class):
        target = y_true[i]
        pred = y_pred[i]
        if target.sum():
            score = 2 * (target * pred).sum() / float((target.sum() + pred.sum()))
            res.append(round(score, decimal))
        else:
            res.append(None)

    return res


def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall


def pad2factor(image, factor=16, pad_value=0):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image


def pad2same_size(imgs):
    H, W = 0, 0
    for img in imgs:
        H = max(H, img.shape[0])
        W = max(W, img.shape[1])

    new_imgs = []
    for img in imgs:
        H_pad, W_pad = H - img.shape[0], W - img.shape[1]
        img = np.pad(img, [[0, H_pad], [0, W_pad]])
        
        new_imgs.append(img)
        
    return new_imgs


def pad2same_size_3d(imgs):
    D, H, W = 0, 0, 0
    for img in imgs:
        D = max(D, img.shape[0])
        H = max(H, img.shape[1])
        W = max(W, img.shape[2])

    new_imgs = []
    for img in imgs:
        D_pad, H_pad, W_pad = D - img.shape[0], H - img.shape[1], W - img.shape[2]
        img = np.pad(img, [[0, D_pad], [0, H_pad], [0, W_pad]])
        
        new_imgs.append(img)
        
    return new_imgs


def normalize(img, minimum=-1024, maximum=3076):
    img = copy.deepcopy(img)

    hir = float(np.percentile(img, 100.0 - 0.5))
    img[img > hir] = hir
    img[img > maximum] = maximum
    img[img < minimum] = minimum
    # 0 ~ 1
    img = (img - minimum) / max(1, (maximum - minimum))
    
    # -1 ~ 1
    img = img * 2 - 1
    return img


def onehot2multi_mask(onehot):
    num_class, D, H, W = onehot.shape
    multi_mask = np.zeros((D, H, W))

    for i in range(1, num_class):
        multi_mask[onehot[i] > 0] = i

    return multi_mask

def load_dicom_image(foldername):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(foldername)
    reader.SetFileNames(dicom_names)
    itkimage = reader.Execute()
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def get_smallest_dcm(path, ext='.dcm'):
    """
    Get smallest dcm file in size given path of target dir
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        ext (str): extension of the DICOM files are defined with
     Return:
        
    """
    fsize_dict = {f:os.path.getsize(path +f) for f in os.listdir(path)}
    for fname, size in [(k, fsize_dict[k]) for k in sorted(fsize_dict, key=fsize_dict.get, reverse=False)]:
        if ext in fname:
            return fname
        
def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence 
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")
    return contour_file

def get_roi_names(contour_data):
    """
    This function will return the names of different contour data, 
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the 
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names
    


def coord2pixels(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return
        img_arr: 2d np.array of image with pixel intensities
        contour_arr: 2d np.array of contour with 0 and 1 labels
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(path + '/CT.'+ img_ID + '.dcm')
    img_arr = img.pixel_array

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((y - origin_y) / y_spacing), np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]
    pixel_coords = [(x, y) for x, y in pixel_coords if x >= 0 and y >= 0 and x < 512 and y < 512]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, img_ID


def cfile2pixels(file, path, ROIContourSeq=0):
    """
    Given a contour file and path of related images return pixel arrays for contours
    and their corresponding images.
    Inputs
        file: filename of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0 (RTV)
    Return
        contour_iamge_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(path + file)
    # index 0 means that we are getting RTV information
    RTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in RTV.ContourSequence]
    img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]  # list of img_arr, contour_arr, im_id

    # debug: there are multiple contours for the same image indepently
    # sum contour arrays and generate new img_contour_arrays
    contour_dict = defaultdict(int)
    for im_arr, cntr_arr, im_id in img_contour_arrays:
        contour_dict[im_id] += cntr_arr
    image_dict = {}
    for im_arr, cntr_arr, im_id in img_contour_arrays:
        image_dict[im_id] = im_arr
    img_contour_arrays = [(image_dict[k], contour_dict[k], k) for k in image_dict]

    return img_contour_arrays


def plot2dcontour(img_arr, contour_arr, figsize=(20, 20)):
    """
    Shows 2d MR img with contour
    Inputs
        img_arr: 2d np.array image array with pixel intensities
        contour_arr: 2d np.array contour array with pixels of 1 and 0
    """

    masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.7)
    plt.show()


def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_contour_dict(contour_file, path, index):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_file: .dcm contour file name
        path: path which has contour and image files
    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path, index)

    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        contour_dict[img_id] = [img_arr, contour_arr]

    return contour_dict

def get_data(path, index):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
        index (int): index of the 
    """
    images = []
    contours = []
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get contour file
    contour_file = get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    contour_dict = get_contour_dict(contour_file, path, index)

    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
        # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(path + k + '.dcm').pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)


def fill_contour(contour_arr):
    H, W = contour_arr.shape

    contour_arr = contour_arr.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = np.zeros((H + 2, W + 2), np.uint8)

    dilation = cv2.dilate(contour_arr, kernel, iterations=1)
    closing = cv2.floodFill(dilation.copy(), mask, (0,0), 1);
    add = np.bitwise_or((1 - closing[1]), dilation)
    erosion = cv2.erode(add, kernel, iterations=1)

    return erosion


def create_image_mask_files(path, index, img_format='png'):
    """
    Create image and corresponding mask files under to folders '/images' and '/masks'
    in the parent directory of path.
    
    Inputs:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        index (int): index of the desired ROISequence
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    X, Y = get_data(path, index)
    Y = np.array([fill_contour(y) if y.max() == 1 else y for y in Y])

    # Create images and masks folders
    new_path = '/'.join(path.split('/')[:-2])
    os.makedirs(new_path + '/images/', exist_ok=True)
    os.makedirs(new_path + '/masks/', exist_ok=True)
    for i in range(len(X)):
        plt.imsave(new_path + '/images/image_{i}.{img_format}', X[i, :, :])
        plt.imsave(new_path + '/masks/mask_{i}.{img_format}', Y[i, :, :])


def ctrdata2pixels(contours, origin, spacing, reso=[512, 512]):
    origin_z, origin_y, origin_x = origin
    spacing_z, spacing_y, spacing_x = spacing
    
    z = []
    contour_arrs = []
    mask_arrs = []
    for i in range(len(contours)):
        contour_coord = contours[i].ContourData
        
        # x, y, z coordinates of the contour in mm
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))
        
        pixel_coords = [(np.ceil((y - origin_y) / spacing_y), np.ceil((x - origin_x) / spacing_x)) for x, y, _ in coord]
        pixel_coords = [(y, x) for x, y in pixel_coords if x >= 0 and y >= 0 and x < 512 and y < 512]
        z.append(int((coord[0][-1] - origin_z) / spacing_z))
        
        mask = np.zeros(reso)
        ctr = np.zeros(reso)
        pixel_coords = np.array([pixel_coords], dtype=np.int32)
        mask = cv2.fillPoly(mask, pixel_coords, color=(1,) * 1)
        
        pts = measure.find_contours(mask, 0)
        if pts:
            # There is contour in the image
            pts = np.concatenate(pts).astype(np.int32)
            for point in pts:
                ctr[point[0], point[1]] = 1
        
        mask_arrs.append(mask)
        contour_arrs.append(ctr)
    
    return z, contour_arrs, mask_arrs
    

def get_patient_data(p_dir):
    contour_fn = get_contour_file(p_dir)
    contour_data = pydicom.dcmread(os.path.join(p_dir, contour_fn))
    img, origin, spacing = load_dicom_image(p_dir)
#     img = truncate_HU_uint8(img)

    i2roi_name = get_roi_names(contour_data)
    rois = {}
    
    D, H, W = img.shape
    contour = np.zeros((len(i2roi_name), D, H, W))
    mask = np.zeros((len(i2roi_name), D, H, W))
    
    for i, name in enumerate(i2roi_name):
        rois[name] = i

    colors = {}
    for roi_name, index in rois.items():
        clr = np.array(cm.hot(float(index) / max(rois.values())))
        clr = clr * 255
        clr [-1] = 255
        colors[roi_name] = clr.astype('uint8')

    for i, name in enumerate(i2roi_name):
        rois[name] = i

    for roi_name in rois.keys():
        index = rois[roi_name]
        if hasattr(contour_data.ROIContourSequence[index], 'ContourSequence'):
            CS = contour_data.ROIContourSequence[index]
            contours = [ctr for ctr in CS.ContourSequence]
            z, contour_arrays, mask_arrays = ctrdata2pixels(contours, origin, spacing, reso=[H, W])

            for i in range(len(z)):
                contour[index][z[i]][contour_arrays[i] > 0] = 1
                mask[index][z[i]][mask_arrays[i] > 0] = 1
        else:
            rois.pop(roi_name, None)
            
    return img, contour, mask, rois, i2roi_name, colors


def get_patient_data_v2(img_dir, ctr_path):
    contour_data = pydicom.dcmread(os.path.join(ctr_path))
    img, origin, spacing = load_dicom_image(img_dir)
#     img = truncate_HU_uint8(img)

    i2roi_name = get_roi_names(contour_data)
    rois = {}
    
    D, H, W = img.shape
    contour = np.zeros((len(i2roi_name), D, H, W))
    mask = np.zeros((len(i2roi_name), D, H, W))
    
    for i, name in enumerate(i2roi_name):
        rois[name] = i

    colors = {}
    for roi_name, index in rois.items():
        clr = np.array(cm.hot(float(index) / max(rois.values())))
        clr = clr * 255
        clr [-1] = 255
        colors[roi_name] = clr.astype('uint8')

    for i, name in enumerate(i2roi_name):
        rois[name] = i

    for roi_name in rois.keys():
        index = rois[roi_name]
        if hasattr(contour_data.ROIContourSequence[index], 'ContourSequence'):
            CS = contour_data.ROIContourSequence[index]
            contours = [ctr for ctr in CS.ContourSequence]
            z, contour_arrays, mask_arrays = ctrdata2pixels(contours, origin, spacing, reso=img.shape[1:])

            for i in range(len(z)):
                contour[index][z[i]][contour_arrays[i] > 0] = 1
                mask[index][z[i]][mask_arrays[i] > 0] = 1
        else:
            rois.pop(roi_name, None)
            
    return img, contour, mask, rois, i2roi_name, colors


def truncate_HU_uint8(img):
    """Truncate HU range and convert to uint8."""

    HU_range = np.array([-1200., 600.])
    new_img = (img - HU_range[0]) / (HU_range[1] - HU_range[0])
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    new_img = (new_img * 255).astype('uint8')
    return new_img
