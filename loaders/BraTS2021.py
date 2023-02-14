from lib import *
from loaders.loader import Loader


class BraTS2021(Loader):
    MODALITIES = ['t1', 't1ce', 't2', 'flair']
    LABELS = [0,1,2,4]
    LBL_LEGENDS = ['Normal', 'Necrotic','Peritumoral', 'Enhancing']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = []

    def get_paths(self):
        src_paths = sorted(list(iglob(join(self.src_dataset_dir, '**', '*.nii.gz'), recursive=True)))
        src_paths_trimmed = []
        dst_paths = []

        for src_path in src_paths:
            filename = basename(src_path).split('.nii.gz')[0]
            uid = '_'.join(filename.split('_')[:2])
            modality = filename.split('_')[2]

            if uid in self.OUTLIERS:
                tqdm.write(f'Skipped outlier: {src_path}')
                continue

            if modality == 'seg':
                dst_paths.append(join(
                    join(self.dst_dataset_dir, 'labelsTr'),
                    f'{uid}.nii.gz',
                ))
            else:
                if modality in self.excluded_mods: continue
                
                dst_paths.append(join(
                    join(self.dst_dataset_dir, 'imagesTr'),
                    f'{uid}_{self.new_alloc_mods.index(modality.lower()):04d}.nii.gz',
                ))
            
            src_paths_trimmed.append(src_path)
        return src_paths_trimmed, dst_paths
    

    def copy(self, src_path, dst_path):
        if '_seg' in basename(src_path).split('.nii.gz')[0][-4:]:
            if self.excluded_labels is None:
                raise NotImplementedError # TODO: 1, 2, 4 -> 1, 2, 3
                tqdm.write(f'{source_path} -> {dest_path.replace("_seg", "").replace("imagesTr", "labelsTr")}')
                src_paths_trimmed.append(src_path)
                dst_paths.append()
                copy(source_path, dest_path.replace('_seg', '').replace('imagesTr', 'labelsTr'))
            else:
                raw = nib.load(src_path)
                rawdata = raw.get_fdata()
                new_lbl = rawdata
                for i in self.excluded_labels:
                    new_lbl = np.where(new_lbl==i, 0, new_lbl)
                for lbl, key in zip(self.remain_labels, self.new_alloc_labels.keys()):
                    if key == 0: continue
                    new_lbl = np.where(new_lbl==lbl, key, new_lbl)
                new = nib.Nifti1Image(new_lbl, header=raw.header, affine=raw.affine) 
                nib.save(new, dst_path)
        else:
            copy(src_path, dst_path)