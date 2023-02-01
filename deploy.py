from lib import *
from utils import generate_dataset_json

DATASET_ROOT_DIR = 'Rawdata/BraTS2021_Training_Data'
DATASET_LOADER = 'loader_BraTS2021'
DST_DIR = './nnUNet'
TASK_NAME = 'Task670_BraTS2021_T1_1Necrotic'
TASK_NAME = 'Task901'
REMAIN_ORIGIN_LABEL = 1
REMOVE_LABELS = [2,4]
REMOVE_MODALITIES = ['t1', 't1ce', 't2']
TASK_IDS = [
    [670, 671, 672, 673],
    [670, 671, 672, 673],
    [670, 671, 672, 673],
]


if __name__ == '__main__':

    loader = find_loader(DATASET_LOADER)(dataset_dir=join(DST_DIR, TASK_NAME))
    source_paths = sorted(list(iglob(join(DATASET_ROOT_DIR, '**', '*.nii.gz'), recursive=True)))[:10]
    dest_paths = [join(DST_DIR, TASK_NAME, 'imagesTr', basename(path)) for path in source_paths]
    run_multiproc(loader.copy, source_paths, dest_paths, desc='Deploying dataset', num_processes=1)
    generate_dataset_json(join(DST_DIR, TASK_NAME, 'dataset.json'), join(DST_DIR, TASK_NAME, 'imagesTr'), None, loader.MODALITIES, {label: legend for label, legend in zip(loader.LABELS, loader.LEGENDS_SHORT)}, dataset_name=TASK_NAME)
    print('Total num of imagesTr: ', len(glob(join(DST_DIR, TASK_NAME, 'imagesTr', '*.nii.gz'))))
    print('Total num of labelsTr: ', len(glob(join(DST_DIR, TASK_NAME, 'labelsTr', '*.nii.gz'))))
