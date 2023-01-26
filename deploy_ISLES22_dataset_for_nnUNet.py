import os, numpy as np, nibabel as nib, SimpleITK as sitk, json, csv, cv2, re, random
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext
from shutil import copy, copytree
from glob import glob, iglob
from sklearn.model_selection import train_test_split
from utils import generate_dataset_json

DATASET_ROOT_DIR = 'ISLES22'
TASK_NAME = 'Task700_ISLES22'
MODALITIES = ['adc', 'dwi', 'flair']
LABELS = [0, 1]
LEGENDS = {i: legend for i, legend in zip(LABELS, ['Normal', 'Label'])}

EXCLUDED_UID = [
    '0216',
    '0224',
    '0157',
    '0150',
    '0170',
    '0240',
    '0052',
    '0122',
]

if __name__ == '__main__':
    
    os.makedirs(join(TASK_NAME, 'imagesTr'), exist_ok=True)
    os.makedirs(join(TASK_NAME, 'labelsTr'), exist_ok=True)

    for path in tqdm(sorted(list(iglob(join(DATASET_ROOT_DIR, '**', '*.nii.gz'), recursive=True)))):
        tqdm.write(path)
        fname = basename(path).split('.nii.gz')[0]
        prefix, suffix = fname.split('_ses-0001_')
        uid = prefix.split('strokecase')[1]
        base_uid = 'ISLES_' + uid

        if uid in EXCLUDED_UID:
            tqdm.write(f'Skipped UID: {path}')
            continue

        if suffix == 'msk':
            copy(path, join(TASK_NAME, 'labelsTr', base_uid + '.nii.gz'))
        else:
            dest_path = '_'.join((base_uid, f'{MODALITIES.index(suffix):04d}'))
            copy(path, join(TASK_NAME, 'imagesTr', dest_path + '.nii.gz'))

    generate_dataset_json(join(TASK_NAME, 'dataset.json'), join(TASK_NAME, 'imagesTr'), None, MODALITIES, LEGENDS, dataset_name=TASK_NAME)
    print('Total num of imagesTr: ', len(glob(join(TASK_NAME, 'imagesTr', '*.nii.gz'))))
    print('Total num of labelsTr: ', len(glob(join(TASK_NAME, 'labelsTr', '*.nii.gz'))))
