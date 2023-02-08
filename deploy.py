from lib import *

# For ISLES
DATASET_ROOT_DIR = 'Rawdata/Task700_ISLES_preprocessed_ADC_DWI_FLAIR'
DATASET_LOADER = 'loader_ISLES22_preprocessed'
# For BraTS
DATASET_ROOT_DIR = 'Rawdata/BraTS2021_Training_Data'
DATASET_LOADER = 'loader_BraTS2021'
DATASET_NAME = 'BraTS2021'

DST_DIR = 'C:/Apache24/htdocs/lab.loopback.kr/datasets/nnUNet'
TASK_IDS = [999, 999, 999, 999]
# TASK_IDS = [670, 671, 672]
# TASK_IDS = [673, 674, 675]
# TASK_IDS = [676, 677, 678]
# TASK_IDS = [630]
EXCLUDED_MODS = [
    ['t1ce', 't2', 'flair'],
    ['t1', 't2', 'flair'],
    ['t1', 't1ce', 'flair'],
    ['t1', 't1ce', 't2'],
]
EXCLUDED_LABELS = [
    [0,1,2],
    [0,1,2],
    [0,1,2],
    [0,1,2],
]


TASK_NAMES = {
    id: f'Task{id}_{DATASET_NAME}_{"_".join(list(set(find_loader(DATASET_LOADER).MODALITIES) - set(excluded_mods))).upper()}_{"_".join([find_loader(DATASET_LOADER).LBL_LEGENDS[i] for i in [find_loader(DATASET_LOADER).LABELS.index(i) for i in list(set(find_loader(DATASET_LOADER).LABELS) - set(excluded_lbls))]])}'
    
    for id, excluded_mods, excluded_lbls in zip(
        TASK_IDS,
        EXCLUDED_MODS,
        EXCLUDED_LABELS,
    )
}


if __name__ == '__main__':
    
    for idx, (task_id, task_name) in enumerate(TASK_NAMES.items()):
    
        loader = find_loader(DATASET_LOADER)(
            dataset_dir=join(DST_DIR, task_name),
            overwrite=True,
            excluded_mods=EXCLUDED_MODS[idx],
            excluded_labels=EXCLUDED_LABELS[idx]
        )
        
        source_paths = sorted(list(iglob(join(DATASET_ROOT_DIR, '**', '*.nii.gz'), recursive=True)))
        dest_paths = [join(DST_DIR, task_name, 'imagesTr', basename(path)) for path in source_paths]
        
        run_multiproc(loader.copy, source_paths, dest_paths, desc='Deploying dataset', num_processes=1)
        generate_dataset_json(
            join(DST_DIR, task_name, 'dataset.json'),
            task_name,
            loader.new_alloc_labels,
            loader.new_alloc_mods,
            join(DST_DIR, task_name, 'imagesTr'),
            None,
        )
        
        print('Total num of imagesTr: ', len(glob(join(DST_DIR, task_name, 'imagesTr', '*.nii.gz'))))
        print('Total num of labelsTr: ', len(glob(join(DST_DIR, task_name, 'labelsTr', '*.nii.gz'))))
