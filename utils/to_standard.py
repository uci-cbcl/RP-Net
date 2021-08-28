import sys
sys.path.append("../")

import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
from multiprocessing import Pool
from visualize import *
from utils.util import *
import nrrd
import traceback


data_dir = '/home/htang6/workspace/data/abdomen/AbodmenNocontrast/inHouse/raw/'
save_dir = '/home/htang6/workspace/data/abdomen/AbodmenNocontrast/standard'

# Use get_patient_data_v2 if the annotation is provided in a seperate folder for each patient
annos_dir = '/home/htang6/workspace/data/abdomen/AbodmenNocontrast/inHouse/raw/'

rois = config['roi_names']


def get_roi_mask(roi_name, roi_names, mask):
    res = np.zeros(mask[0].shape)
    for n in roi_names.keys():
        if 'prv' in n.lower():
            print('found prv in this %s' % (n))
            continue
        if roi_name.lower() in n.lower():
            res = np.logical_or(res, mask[roi_names[n]])
    
    return res.astype(np.uint8)


def process_patient(param):
    # Read in raw dicom file and dicom RS
    # Save the image and annotation as the same format as MICCAI15 challenge (PDDCA)
    pid = param
    try:
        print('processing ', os.path.join(data_dir, pid))
        # img, contour, mask, roi_names, i2roi_name, colors = get_patient_data(os.path.join(data_dir, pid))
        img, contour, mask, roi_names, i2roi_name, colors = get_patient_data_v2(os.path.join(data_dir, pid, 'CT'),
                                                                                os.path.join(annos_dir, pid, 'RS_gt', os.listdir(os.path.join(annos_dir, pid, 'RS_gt'))[0]))

        if not os.path.exists(os.path.join(save_dir, pid, 'structures')):
            os.makedirs(os.path.join(save_dir, pid, 'structures'))

        img = np.swapaxes(img, 0, -1).astype(np.float32)
        nrrd.write(os.path.join(save_dir, pid, 'img.nrrd'), img)

        for roi_name in rois:
            m = get_roi_mask(roi_name, roi_names, mask)
            m = np.swapaxes(m, 0, -1)
            if np.any(m):
                nrrd.write(os.path.join(save_dir, pid, 'structures', '%s.nrrd' % (roi_name)), m)
        print('Finished processing patient ', os.path.join(data_dir, pid))
    except Exception as e:
        print('Caught exception in preprocessing %s:' % (os.path.join(data_dir, pid)))
        traceback.print_exc()
        
        print()


def main():
    params = []

    for pid in os.listdir(data_dir):
        params.append((pid))
    print('Total # of cases ', len(params))

    pool = Pool(processes=4)
    pool.map(process_patient, params)
    
    pool.close()
    pool.join()
    
    
if __name__ == '__main__':
    main()
