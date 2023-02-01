import os, numpy as np, nibabel as nib, SimpleITK as sitk, json, csv, cv2, re, random
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext, dirname
from shutil import copy, copytree
from glob import glob, iglob
from sklearn.model_selection import train_test_split


class Loader_BraTS2021:
    MODALITIES = ['t1', 't1ce', 't2', 'flair']
    LEGENDS = ['Normal', 'Necrotic_tumor_core','Peritumoral_invaded_tissue','GD_enhancing_tumor']
    LEGENDS_SHORT = ['Normal', 'Necrotic','Peritumoral','Enhancing']
    LABELS = [0,1,2,4]
    OUTLIERS = ['BraTS2021_01666']

    def __init__(
        self,
        dataset_dir,
        remove_modalities: list=None,
        remove_labels: list=None,
        remain_label: int=None,
        ) -> None:
        self.remove_modalities = remove_modalities
        self.remove_labels = remove_labels
        self.remain_label = remain_label
        os.makedirs(join(dataset_dir, 'imagesTr'), exist_ok=True)
        os.makedirs(join(dataset_dir, 'labelsTr'), exist_ok=True)

    def copy(self, source_path, dest_path):
        if basename(source_path) in self.OUTLIERS:
            tqdm.write(f'Skipped outlier: {source_path}')
            return
        tqdm.write(f'{source_path} -> {dest_path}')

        uid = '_'.join(basename(source_path).split('.nii.gz')[0].split('_')[:2])
        modality = basename(source_path).split('.nii.gz')[0].split('_')[2]

        if modality == 'seg':
            if self.remove_labels is None:
                copy(source_path, dest_path.replace('_seg', '').replace('imagesTr', 'labelsTr'))
            else:
                raw = nib.load(source_path)
                rawdata = raw.get_fdata()
                new_lbl = rawdata
                for i in self.remove_labels:
                    new_lbl = np.where(new_lbl==i, 0, new_lbl)
                new_lbl = np.where(new_lbl==self.remain_label, 1, new_lbl)
                new = nib.Nifti1Image(new_lbl, header=raw.header, affine=raw.affine) 
                nib.save(new, dest_path)
        else:
            dest_dir = dirname(dest_path)
            if self.remove_modalities is None:
                copy(source_path, join(dest_dir, f'{uid}_{self.MODALITIES.index(modality):04d}.nii.gz'))
            else:
                if modality in self.remove_modalities:
                    tqdm.write(f'Skipped modality: {source_path}:{modality}')
                    return
                uid = '_'.join((uid, f'{MODALITIES.index(name_spl[2]):04d}'))
                copy(source_path, join(DST_DIR, TASK_NAME, 'imagesTr', uid + '.nii.gz'))