import os, numpy as np, nibabel as nib, SimpleITK as sitk, json, csv, cv2, re, random
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext, dirname
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from sklearn.model_selection import train_test_split


class Loader_ISLES22_preprocessed:
    MODALITIES = ['adc', 'dwi', 'flair']
    LEGENDS = ['Normal', 'Label']
    LEGENDS_SHORT = LEGENDS
    LABELS = [0, 1]
    OUTLIERS = []

    def __init__(
        self,
        dataset_dir,
        overwirte: bool=False,
        remove_modalities: list=None,
        remove_labels: list=None,
        remain_label: int=None,
        ) -> None:
        self.remove_modalities = remove_modalities
        self.remove_labels = remove_labels
        self.remain_label = remain_label
        if overwirte: rmtree(dataset_dir)
        os.makedirs(join(dataset_dir, 'imagesTr'), exist_ok=True)
        os.makedirs(join(dataset_dir, 'labelsTr'), exist_ok=True)

    def copy(self, source_path, dest_path):
        if basename(source_path) in self.OUTLIERS:
            tqdm.write(f'Skipped outlier: {source_path}')
            return
        tqdm.write(f'{source_path} -> {dest_path}') # TODO: labelTr에 대해서 오류 처리

        if 'labelsTr' in dirname(source_path):
            copy(source_path, dest_path.replace('imagesTr', 'labelsTr'))
        else:
            if self.remove_modalities is None:
                copy(source_path, dest_path)
            else:
                pass