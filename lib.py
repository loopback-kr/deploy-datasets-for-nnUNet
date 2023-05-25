import os, importlib, numpy as np, nibabel as nib, json, csv, re, random, pandas as pd
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext, isdir, dirname
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from sklearn.model_selection import train_test_split
from datetime import datetime
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def find_loader(loader_name: str):
    datasetlib = importlib.import_module(f'loaders.{loader_name}')
    for name, cls in datasetlib.__dict__.items():
        if name == loader_name:
            return cls
    raise NotImplementedError('Unknown loader name')

def run_multiproc(func, *args, desc='', num_processes=os.cpu_count()):
    with tqdm(total=len(args[0]),desc=desc, colour="green", dynamic_ncols=True) as pbar:
        with mpp.Pool(num_processes) as pool:
            for _ in pool.istarmap(func, zip(*args)): # TODO: 리스트가 1개밖에 없는건 자동으로 늘리게 하는 코드 내장하도록
                pbar.update()
            pool.close()
            pool.join()


def generate_dataset_json(output_file: str, dataset_name: str, labels: dict, modalities: tuple, imagesTr_dir: str, imagesTs_dir: str, sort_keys=True, dataset_description: str = "", extension='.nii.gz'):
    train_identifiers = ['_'.join(os.path.basename(path).split(extension)[0].split('_')[:-1]) for path in sorted(list(iglob(os.path.join(imagesTr_dir, '**', f'*{extension}'), recursive=True)))]

    if imagesTs_dir:
        test_identifiers = ['_'.join(os.path.basename(path).split(extension)[0].split('_')[:-1]) for path in sorted(list(iglob(os.path.join(imagesTs_dir, '**', f'*{extension}'), recursive=True)))]
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': f"./imagesTr/{i}{extension}", "label": f"./labelsTr/{i}{extension}"} for i
        in
        train_identifiers]
    json_dict['test'] = [f"./imagesTs/{i}{extension}" for i in test_identifiers]

    with open(os.path.join(output_file), 'w') as f:
        json.dump(json_dict, f, sort_keys=sort_keys, indent=4)
