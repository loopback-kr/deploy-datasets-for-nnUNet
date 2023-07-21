# Python built-in libs for handling filesystems
import sys, os, json, pickle, csv, re, random, logging, importlib, argparse
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
from pathlib import Path
from shutil import copy, copytree, rmtree
from copy import deepcopy
from glob import glob, iglob
# Python utility packages for productivity
from tqdm import tqdm, trange
from tqdm.contrib import tzip
# Datascience packages for medical imaging
import numpy as np, scipy, pandas, cv2 as cv, SimpleITK as sitk, nibabel as nib, nrrd
import matplotlib.pyplot as plt
import nrrd

SRC_DATA_DIR = "First_Align_Generic"
DST_DATA_DIR = "submit"

os.makedirs(DST_DATA_DIR, exist_ok=True)
for path in tqdm(sorted(glob(join(SRC_DATA_DIR, "*.nii.gz")))):
    raw = nib.load(path)
    aligned = raw.get_fdata()
    aligned = np.flip(aligned, axis=1)
    aligned = np.flip(aligned, axis=0)
    aligned_data = nib.Nifti1Image(aligned.astype(np.uint8), None)
    nib.save(aligned_data, join(DST_DATA_DIR, basename(path).replace("DATA", "MASK")))
