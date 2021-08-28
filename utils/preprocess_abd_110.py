import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import os
import SimpleITK as sitk
import nrrd
from multiprocessing import Pool
from utils.util import annotation2multi_mask, annotation2masks, load_dicom_image
from utils.preprocess_pancreas import resample


roi_names = ['Large Bowel', 'Duodenum', 'Spinal Cord', 'Liver', 'Spleen', 'Small Bowel', 'Pancreas', 'Kidney L', 'Kidney R', 'Stomach', 'Gallbladder']
raw_dir = '/home/htang6/workspace/data/abdomen/AbodmenNocontrast/inHouse/raw/'
data_dir = '/home/htang6/workspace/data/abdomen/AbodmenNocontrast/standard'
save_dir = '/home/htang6/workspace/data/abdomen/preprocessed'
# z_starts = get_z_starts(data_dir, pids)
z_starts = None
do_resample = False


def morphology_process(itk_img, radius=7):
    """
    First use threshold to get rough brain region, then
    use morphology closing and opening to remove region outside the brain
    """
    connected_img = 1 - sitk.OtsuThreshold(itk_img)
    closed_img = sitk.BinaryMorphologicalClosing(connected_img, radius)
    opened_img = sitk.BinaryMorphologicalOpening(closed_img, radius)

    H, W = sitk.GetArrayFromImage(itk_img).shape
    seed = [(H // 2, W // 2)]
    mask_img = sitk.ConnectedThreshold(opened_img, seedList=seed, lower=1)
    mask_img = sitk.BinaryFillhole(mask_img)

    return mask_img

def preprocess_image(itk_img):
    """
    Preprocess itk image slice by slice
    """
    width, height, depth = itk_img.GetWidth(), itk_img.GetHeight(), itk_img.GetDepth()
    npy_mask = np.zeros((depth, height, width))
    for i in range(depth):
        npy_mask[i, :, :] = sitk.GetArrayFromImage(morphology_process(itk_img[:, :, i]))

    return npy_mask


def main():
    pids = os.listdir(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    pool = Pool(processes=4)
    pool.map(preprocess, pids)

    pool.close()
    pool.join()

def preprocess(params):
    pid = params
    image, meta = nrrd.read(os.path.join(data_dir, pid, 'img.nrrd'))
    image = np.swapaxes(image, 0, -1)

    new_spacing = np.array([2., 2., 2.])
    _, _, spacing = load_dicom_image(os.path.join(raw_dir, pid, 'CT'))
    
    if do_resample:
        print('resampling', spacing, new_spacing)
        image, _ = resample(image, spacing, new_spacing)

    processed_image = image.copy()

    if z_starts is not None:
        z_start = z_starts[pid]
    else:
        z_start = 0
    processed_image = processed_image[z_start:, :, :]

    # Get binary mask for brain region, remove human hair and other tissues
    npy_mask = preprocess_image(sitk.GetImageFromArray(processed_image))
    processed_image[npy_mask == 0] = -1024

    # Crop only brain region to reduce image size
    _, yy, xx = np.where(processed_image > -1024)
    y_start = yy.min()
    y_end = yy.max()
    x_start = xx.min()
    x_end = xx.max()
    processed_image = processed_image[:, y_start:y_end, x_start:x_end]

    bbox = np.array([[z_start, y_start, x_start], [z_start + image.shape[0], y_end, x_end]])
    np.save(os.path.join(save_dir, '%s_raw.npy' % (pid)), image)
    np.save(os.path.join(save_dir, '%s_bbox.npy' % (pid)), bbox)
    nrrd.write(os.path.join(save_dir, '%s_clean.nrrd' % (pid)), processed_image)
    print(pid, ' ', processed_image.shape)

    masks = {}
    for roi in roi_names:
        if os.path.isfile(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi))):
            mask, meta = nrrd.read(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi)))
            mask = np.swapaxes(mask, 0, -1)

            if do_resample:
                mask, _ = resample(mask, spacing, new_spacing)
                mask = mask > 0.5

            mask = mask[:, y_start:y_end, x_start:x_end]
            mask = mask.astype(np.uint8)
            masks[roi] = mask
            nrrd.write(os.path.join(save_dir, '%s_%s.nrrd' % (pid, roi)), mask)

    masks = annotation2masks(masks).astype(np.uint8)
    nrrd.write(os.path.join(save_dir, '%s_masks.npy' % (pid)), masks)
    # np.save(os.path.join(save_dir, '%s_masks.npy' % (pid)), masks)


def get_z_starts(data_dir, pids):
    z_starts = {}
    for pid in pids:
        min_z = np.inf
        for roi in roi_names:
            if os.path.isfile(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi))):
                mask, meta = nrrd.read(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi)))
                mask = np.swapaxes(mask, 0, -1)
                min_z = min(min_z, np.where(mask > 0)[0][0])

    #             if crop_slice_idx[pid] > np.where(mask > 0)[0][0]:
    #                 print '%s wrong z slice start %d, %d' % (pid, crop_slice_idx[pid], np.where(mask > 0)[0][0])
            else:
                print('%s does not have %s' % (pid, roi))

        print('%s, min_z %d' % (pid, min_z))
        z_starts[pid] = min_z - 4

    return z_starts


if __name__ == '__main__':
    main()