import os, numpy as np, nibabel as nib, SimpleITK as sitk, json, csv, cv2, re, random
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext
from shutil import copy, copytree
from glob import glob, iglob
from sklearn.model_selection import train_test_split
from utils import generate_dataset_json

DATASET_ROOT_DIR = 'BraTS2021_Training_Data'
TASK_NAME = 'Task600_BraTS2021'
MODALITIES = ['t1', 't1ce', 't2', 'flair']
LABELS = [0,1,2,4]
LEGENDS = {i: legend for i, legend in zip(LABELS, ['Brain MRI', 'Necrotic tumor core','Peritumoral invaded tissue','GD-enhancing tumor'])}

if __name__ == '__main__':
    
    os.makedirs(join(TASK_NAME, 'imagesTr'), exist_ok=True)
    os.makedirs(join(TASK_NAME, 'labelsTr'), exist_ok=True)

    for path in tqdm(sorted(list(iglob(join(DATASET_ROOT_DIR, '**', '*.nii.gz'), recursive=True)))):
        tqdm.write(path)
        fname = basename(path)
        name = fname.split('.nii.gz')[0]
        name_spl = name.split('_')
        
        base_uid = '_'.join(name_spl[:2])

        if name_spl[2] == 'seg':
            copy(path, join(TASK_NAME, 'labelsTr', uid + '.nii.gz'))
        else:
            uid = '_'.join((base_uid, f'{MODALITIES.index(name_spl[2]):04d}'))
            copy(path, join(TASK_NAME, 'imagesTr', uid + '.nii.gz'))

    generate_dataset_json(join(TASK_NAME, 'dataset.json'), join(TASK_NAME, 'imagesTr'), None, MODALITIES, LEGENDS, dataset_name=TASK_NAME)
    print('Total num of imagesTr: ', len(glob(join(TASK_NAME, 'imagesTr', '*.nii.gz'))))
    print('Total num of labelsTr: ', len(glob(join(TASK_NAME, 'labelsTr', '*.nii.gz'))))
