# Python built-in libs for handling filesystems
import sys, os, json, pickle, csv, re, random, logging, importlib, argparse
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
from pathlib import Path
from shutil import copy, copytree, rmtree
from copy import deepcopy
from glob import glob, iglob
# Datascience packages for medical imaging
import numpy as np, scipy, pandas, cv2 as cv, SimpleITK as sitk, nibabel as nib, nrrd
import matplotlib.pyplot as plt
# Python utility packages for productivity
from tqdm import tqdm, trange
from tqdm.contrib import tzip

data, hdr = nrrd.read("MASK_000.nrrd")
print(np.unique(data), data.dtype)


for path in sorted(glob(join("submit/*.nii.gz"))):
    raw = nib.load(path)
    data = raw.get_fdata()
    # print(raw.header)
    print(np.unique(data), data.dtype)