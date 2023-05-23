import nrrd, pandas as pd
from lib import *
from loaders.loader import Loader

class TDSC2023(Loader):
    MODALITIES = ['NRRD_ALIGN']
    LABELS = [0, 1, 2]
    LBL_LEGENDS = ['Normal', 'Benign', 'Malignant']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = []#['094', '098', '099', '095', '096', '097']
    EXTENSION = '.nrrd'

    def __init__(
        self,
        src_dataset_dir,
        dst_dataset_dir,
        overwrite: bool=False,
        excluded_mods: list=None,
        excluded_labels: list=None,
        format: str=EXTENSION,
        dirs_ratio: dict={'tr': 0.7, 'val': 0.2, 'ts': 0.1},
        ) -> None:
        super().__init__(src_dataset_dir, dst_dataset_dir, overwrite, excluded_mods, excluded_labels, format, dirs_ratio)

        self.df = pd.read_csv(join(self.src_dataset_dir, 'labels.csv'))
    
    def get_paths(self):
        
        img_paths = sorted(list(iglob(join(self.src_dataset_dir, 'DATA*', f'*{self.EXTENSION}'), recursive=True)))
        lbl_paths = sorted(list(iglob(join(self.src_dataset_dir, 'MASK', f'*{self.EXTENSION}'), recursive=True)))
        assert len(img_paths) == len(lbl_paths)
        
        src_paths_wo_outliers = []
        dst_paths = []

        ids = [basename(path).split(self.EXTENSION)[0] for path in lbl_paths]

        train_ids, valid_test_ids = train_test_split(ids, test_size=np.trunc((1-self.dirs_ratio['tr'])*100000)/100000, shuffle=True, random_state=0)
        valid_ids, test_ids = train_test_split(valid_test_ids, test_size=np.trunc((self.dirs_ratio['val']+self.dirs_ratio['ts'])*100000)/100000, shuffle=True, random_state=0)

        for img_path, lbl_path in zip(img_paths, lbl_paths):
            id = basename(lbl_path).split(self.EXTENSION)[0]
            if id in self.OUTLIERS:
                continue
            assert id.split('_')[1] == basename(img_path).split(self.EXTENSION)[0].split('_')[1]

            dst_img_path = join(self.dst_dataset_dir, 'imagesTr' if id in train_ids else 'imagesVal' if id in valid_ids else 'imagesTs', basename(img_path))
            dst_lbl_path = join(self.dst_dataset_dir, 'labelsTr' if id in train_ids else 'labelsVal' if id in valid_ids else 'labelsTs', basename(lbl_path))

            src_paths_wo_outliers.append(img_path)
            src_paths_wo_outliers.append(lbl_path)
            dst_paths.append(dst_img_path)
            dst_paths.append(dst_lbl_path)

        return src_paths_wo_outliers, dst_paths
    
    def copy(self, src_path, dst_path):
        data, header = nrrd.read(src_path)
        
        label = self.df[['ID', 'label']].where(self.df['ID'] == basename(src_path).replace('MASK', 'DATA')).dropna()['label'].to_string(index=False)
        if label == 'M':
            # data[data == 1] = 2
            # dst_path = dst_path.replace('DATA', 'Malignant')
            header['label'] = 'malignant'
        else:
            # dst_path = dst_path.replace('DATA', 'Benign')
            header['label'] = 'benign'

        if self.format == '.nrrd':
            aligned = data
            aligned = np.rot90(aligned, k=2, axes=(0, 1))
            aligned = np.flip(aligned, axis=1)
            aligned = np.flip(aligned, axis=0)
            nrrd.write(dst_path, aligned, header)
        elif self.format == '.nii.gz':
            aligned = data
            aligned = np.flip(aligned, axis=1)
            aligned = np.flip(aligned, axis=0)
            img = nib.Nifti1Image(aligned, np.eye(4), header=header)
            # assert np.all(data == img.get_fdata())
            # ax = plt.subplot(3, 1, 1)
            # ax = plt.subplot(3, 1, 3)
            # ax.imshow(img.get_fdata()[:, :, 150], cmap='gray')
            nib.save(img, dst_path)
        else:
            raise Exception
        
        # ax = plt.subplot(3, 1, 2)
        # ax.imshow(aligned[:, :, 150], cmap='gray')
        # ax.imshow(data[:, :, 150], cmap='gray')
        # plt.show()