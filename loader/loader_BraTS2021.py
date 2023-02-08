from lib import *


class Loader_BraTS2021:
    MODALITIES = ['t1', 't1ce', 't2', 'flair']
    LABELS = [0,1,2,4]
    LBL_LEGENDS = ['Normal', 'Necrotic','Peritumoral', 'Enhancing']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = ['BraTS2021_01666']

    def __init__(
        self,
        dataset_dir,
        overwrite: bool=False,
        excluded_mods: list=None,
        excluded_labels: list=None,
        ) -> None:
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

        if isdir(dataset_dir) and overwrite: rmtree(dataset_dir)
        os.makedirs(join(dataset_dir, 'imagesTr'), exist_ok=True)
        os.makedirs(join(dataset_dir, 'labelsTr'), exist_ok=True)

    def copy(self, source_path, dest_path):
        if basename(source_path) in self.OUTLIERS:
            tqdm.write(f'Skipped outlier: {source_path}')
            return

        uid = '_'.join(basename(source_path).split('.nii.gz')[0].split('_')[:2])
        modality = basename(source_path).split('.nii.gz')[0].split('_')[2]

        if modality == 'seg':
            if self.excluded_labels is None:
                tqdm.write(f'{source_path} -> {dest_path.replace("_seg", "").replace("imagesTr", "labelsTr")}')
                copy(source_path, dest_path.replace('_seg', '').replace('imagesTr', 'labelsTr'))
            else:
                raw = nib.load(source_path)
                rawdata = raw.get_fdata()
                new_lbl = rawdata
                for i in self.excluded_labels:
                    new_lbl = np.where(new_lbl==i, 0, new_lbl)
                for lbl, key in zip(self.remain_labels, self.new_alloc_labels.keys()):
                    if key == 0: continue
                    new_lbl = np.where(new_lbl==lbl, key, new_lbl)
                new = nib.Nifti1Image(new_lbl, header=raw.header, affine=raw.affine) 
                tqdm.write(f'{source_path} -> {dest_path.replace("_seg", "").replace("imagesTr", "labelsTr")}')
                nib.save(new, dest_path.replace('_seg', '').replace('imagesTr', 'labelsTr'))
        else:
            dest_dir = dirname(dest_path)
            if self.excluded_mods is None:
                tqdm.write(f'{source_path} -> {join(dest_dir, f"{uid}_{self.MODALITIES.index(modality):04d}.nii.gz")}')
                copy(source_path, join(dest_dir, f'{uid}_{self.MODALITIES.index(modality):04d}.nii.gz'))
            else:
                if modality in self.excluded_mods:
                    tqdm.write(f'Skipped modality: {source_path}:{modality}')
                    return
                tqdm.write(f'{source_path} -> {join(dest_dir, f"{uid}_{self.MODALITIES.index(modality):04d}.nii.gz")}')
                copy(source_path, join(dest_dir, f'{uid}_{self.new_alloc_mods.index(modality):04d}.nii.gz'))
