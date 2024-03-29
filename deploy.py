from lib import *


# User defined consts
DATASET_ROOT_DIR = 'raw/TDSC_2023_MICCAI'
DATASET_NAME = 'TDSC2023'
DST_DIR = 'out'
EXCLUDED_MODS = [
    [],
]
EXCLUDED_LABELS = [
    [0],
]

# Generated consts
TASK_NAMES = [
    f'{DATASET_NAME}_{"_".join(list(set(find_loader(DATASET_NAME).MODALITIES) - set(excluded_mods))).upper()}_{"_".join([find_loader(DATASET_NAME).LBL_LEGENDS[i] for i in [find_loader(DATASET_NAME).LABELS.index(i) for i in list(set(find_loader(DATASET_NAME).LABELS) - set(excluded_lbls))]])}'
    for excluded_mods, excluded_lbls in zip(EXCLUDED_MODS, EXCLUDED_LABELS)
]
DIRS_RATIO = {
    'Tr': 70,
    'Val': 20,
    'Ts': 10,
}
RND_SEED = 0


if __name__ == '__main__':
    random.seed(RND_SEED)

    [print(task_name) for task_name in TASK_NAMES]
    
    for idx, task_name in enumerate(TASK_NAMES):
    
        loader = find_loader(DATASET_NAME)(
            src_dataset_dir=join(DATASET_ROOT_DIR),
            dst_dataset_dir=join(DST_DIR, task_name),
            overwrite=True,
            excluded_mods=EXCLUDED_MODS[idx],
            excluded_labels=EXCLUDED_LABELS[idx],
            dirs_ratio=DIRS_RATIO,
        )
        src_phase_paths, dst_phase_paths = loader.get_paths()

        for dir in list(set([dirname(path) for path in dst_phase_paths])):
            os.makedirs(dir, exist_ok=True)
        run_multiproc(loader.copy, src_phase_paths, dst_phase_paths, desc='Deploying dataset', num_processes=min(os.cpu_count(), 99))

    # generate_dataset_json(
    #     join(DST_DIR, task_name, 'dataset.json'),
    #     task_name,
    #     loader.new_alloc_labels,
    #     loader.new_alloc_mods,
    #     join(DST_DIR, task_name, 'imagesTr'),
    #     None,
    # )
    
    # print('Total num of imagesTr: ', len(glob(join(DST_DIR, task_name, 'imagesTr', '*.nii.gz'))))
    # print('Total num of labelsTr: ', len(glob(join(DST_DIR, task_name, 'labelsTr', '*.nii.gz'))))
