import nrrd, pandas as pd
from lib import *
from loaders.loader import Loader


class TDSCABUS2023(Loader):
    MODALITIES = ['NRRD_ALIGN']
    LABELS = [0, 1, 2]
    LBL_LEGENDS = ['Normal', 'Benign', 'Malignant']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = []#['094', '098', '099', '095', '096', '097']

    def __init__(
        self,
        src_dataset_dir,
        dst_dataset_dir,
        target_extension=".nrrd"
        ) -> None:
        self.src_dataset_dir = src_dataset_dir
        self.dst_dataset_dir = dst_dataset_dir
        self.target_extension = target_extension
        self.spec = Loader.DecathlonSpec()
        self.spec.name = 'TDSC-ABUS2023'
        self.df = pd.read_csv(join(self.src_dataset_dir, 'labels.csv'))
    
    def get_paths(self):
        img_paths = sorted(list(iglob(join(self.src_dataset_dir, 'DATA*_*', '*.nrrd'), recursive=True)))
        val_paths = sorted(list(iglob(join(self.src_dataset_dir, 'DATA', '*.nrrd'), recursive=True)))
        lbl_paths = sorted(list(iglob(join(self.src_dataset_dir, 'MASK', '*.nrrd'), recursive=True)))
        assert len(img_paths) == len(lbl_paths)
        
        src_paths = []
        dst_paths = []

        for img_path, lbl_path in zip(img_paths, lbl_paths):
            dst_img_path = join(self.dst_dataset_dir, 'imagesTr', basename(img_path).replace('.nrrd', "_0000" + self.target_extension))
            dst_lbl_path = join(self.dst_dataset_dir, 'labelsTr', basename(lbl_path).replace("MASK", "DATA").replace('.nrrd', self.target_extension))
            src_paths.append(img_path)
            src_paths.append(lbl_path)
            dst_paths.append(dst_img_path)
            dst_paths.append(dst_lbl_path)
        
        for val_path in val_paths:
            dst_val_path = join(self.dst_dataset_dir, 'imagesVal', basename(val_path).replace('.nrrd', "_0000" + self.target_extension))
            src_paths.append(val_path)
            dst_paths.append(dst_val_path)
        
        return src_paths, dst_paths
    
    def copy(self, src_path, dst_path, kwargs:dict={'write_label': False, 'align': False}):
        os.makedirs(dirname(dst_path), exist_ok=True)
        data, header = nrrd.read(src_path)
        
        if kwargs['write_label']:
            label = self.df[['ID', 'label']].where(self.df['ID'] == basename(src_path).replace('MASK', 'DATA')).dropna()['label'].to_string(index=False)
            if label == 'M':
                header['label'] = 'malignant'
            else:
                header['label'] = 'benign'
        
        if self.target_extension == '.nrrd':
            if kwargs['align']:
                aligned = data
                aligned = np.rot90(aligned, k=2, axes=(0, 1))
                aligned = np.flip(aligned, axis=1)
                aligned = np.flip(aligned, axis=0)
                nrrd.write(dst_path, aligned, header)
            else:
                nrrd.write(dst_path, data, header)

        elif self.target_extension == '.nii.gz':
            if kwargs['align']:
                aligned = data
                aligned = np.flip(aligned, axis=1)
                aligned = np.flip(aligned, axis=0)
                img = nib.Nifti1Image(aligned, None)
                nib.save(img, dst_path)
            else:
                img = nib.Nifti1Image(data, None)
                nib.save(img, dst_path)
        else:
            raise Exception

if __name__ == '__main__':
    dataset = TDSCABUS2023(
        src_dataset_dir="D:/datasets/TDSC-ABUS2023",
        dst_dataset_dir="out/TDSC-ABUS2023_NIFTI",
        target_extension=".nii.gz",
    )

    src_paths, dst_paths = dataset.get_paths()
    run_multiproc(
        dataset.copy,
        src_paths,
        dst_paths,
        [{'write_label': False, 'align': False}] * len(src_paths),
    )

    generate_dataset_json(
        output_file=join("out/TDSC-ABUS2023_NIFTI", 'dataset.json'),
        dataset_name="TDSC-ABUS2023",
        labels={0: "background", 1: "label"},
        modalities=["Tumor"],
        imagesTr_dir="out/TDSC-ABUS2023_NIFTI/imagesTr",
        imagesTs_dir=None
    )