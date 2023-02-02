from lib import *

# For ISLES
DATASET_ROOT_DIR = 'Rawdata/Task700_ISLES_preprocessed_ADC_DWI_FLAIR'
DATASET_LOADER = 'loader_ISLES22_preprocessed'
# For BraTS
DATASET_ROOT_DIR = 'Rawdata/BraTS2021_Training_Data'
DATASET_LOADER = 'loader_BraTS2021'
DATASET_NAME = 'BraTS2021'

DST_DIR = './nnUNet'
# TASK_IDS = range(670, 673)
# TASK_IDS = range(673, 676)
# TASK_IDS = range(676, 679)
TASK_IDS = range(679, 682)
EXCLUDED_MODS = [
    ['t1', 't1ce', 't2'],
    ['t1', 't1ce', 't2'],
    ['t1', 't1ce', 't2'],
]
EXCLUDED_LABELS = [
    [0,2,4],
    [0,1,4],
    [0,1,2],
]


TASK_NAMES = {
    id: f'Task{id}-{DATASET_NAME}-{"_".join(list(set(find_loader(DATASET_LOADER).MODALITIES) - set(excluded_mods)))}-{"_".join([find_loader(DATASET_LOADER).LBL_LEGENDS[i] for i in [find_loader(DATASET_LOADER).LABELS.index(i) for i in list(set(find_loader(DATASET_LOADER).LABELS) - set(excluded_lbls))]])}'
    
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
        loader.generate_dataset_json(
            join(DST_DIR, task_name, 'dataset.json'),
            join(DST_DIR, task_name, 'imagesTr'),
            None,
            dataset_name=task_name
        )
        
        print('Total num of imagesTr: ', len(glob(join(DST_DIR, task_name, 'imagesTr', '*.nii.gz'))))
        print('Total num of labelsTr: ', len(glob(join(DST_DIR, task_name, 'labelsTr', '*.nii.gz'))))
