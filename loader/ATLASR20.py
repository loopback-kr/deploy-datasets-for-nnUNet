from lib import *


class ATLASR20:
    MODALITIES = ['t1w']
    LABELS = [0,1]
    LBL_LEGENDS = ['Normal', 'Label']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = ['sub-r039s002']

    def __init__(
        self,
        src_dataset_dir,
        dst_dataset_dir,
        overwrite: bool=False,
        excluded_mods: list=None,
        excluded_labels: list=None,
        ) -> None:
        self.src_dataset_dir = src_dataset_dir
        self.dst_dataset_dir = dst_dataset_dir
        self.excluded_mods = excluded_mods
        if excluded_mods:
            self.new_alloc_mods = list(set(self.MODALITIES) - set(excluded_mods))
        else:
            self.new_alloc_mods = self.MODALITIES
        self.excluded_labels = excluded_labels
        if excluded_labels:
            self.remain_labels = [0] + list(set(self.LABELS_ALT.keys()) - set(excluded_labels))
        else:
            self.remain_labels = [0] + [i for i, _ in enumerate(self.LABELS_ALT.keys()) if i != 0]
        self.new_alloc_labels = {
            i: self.LABELS_ALT[lbl]
            for i, lbl in enumerate(
                self.remain_labels,
            )
        }
        if isdir(dst_dataset_dir) and overwrite: rmtree(dst_dataset_dir)
    
    def get_paths(self):
        src_paths = sorted(list(iglob(join(self.src_dataset_dir, '**', '*.nii.gz'), recursive=True)))
        src_paths_trimmed = []
        dst_paths = []

        for src_path in src_paths:
            filedir = dirname(src_path)
            filename = basename(src_path).split('.nii.gz')[0]
            splitted = filename.split('_ses-1_space-MNI152NLin2009aSym_')
            is_train = True if 'Training' in filedir else False

            # Skip outliers
            if not len(splitted) == 2: continue
            else:
                uid, modality = splitted
            if not modality.lower() in self.MODALITIES + ['label-L_desc-T1lesion_mask'.lower()]: continue
            if uid in self.OUTLIERS: continue
            
            src_paths_trimmed.append(src_path)
            if modality == 'label-L_desc-T1lesion_mask':
                dst_filedir = join(self.dst_dataset_dir, 'labelsTr')
                dst_filename = f'{uid}.nii.gz'
            else:
                dst_filedir = join(self.dst_dataset_dir, 'imagesTr' if is_train else 'imagesTs')
                dst_filename = f'{uid}_{self.MODALITIES.index(modality.lower()):04d}.nii.gz'
            dst_paths.append(join(dst_filedir, dst_filename))
        return src_paths_trimmed, dst_paths
