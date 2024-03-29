from lib import *

class Loader:
    MODALITIES = ['MODALITY']
    LABELS = [0, 1]
    LBL_LEGENDS = ['Normal', 'Label']
    LABELS_ALT = {lbl:legend for lbl, legend in zip(LABELS, LBL_LEGENDS)}
    OUTLIERS = []
    EXTENSION = '.nii.gz'

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
        self.src_dataset_dir = src_dataset_dir
        self.dst_dataset_dir = dst_dataset_dir
        if isdir(dst_dataset_dir) and overwrite: rmtree(dst_dataset_dir)
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
        self.format = format
        self.dirs_ratio = dirs_ratio
    
    def get_paths(self):
        pass
    
    def copy(self, src_path, dst_path):
        copy(src_path, dst_path)
    
    class DecathlonSpec:
        def __init__(self) -> None:
            self.name = 'Dataset name'
            self.description = 'Dataset description'
            self.reference = 'Dataset reference'
            self.release = 'Dataset release'
            self.license = 'Dataset license'
            self.file_ending = 'Data file extension'
            self.modality = 'Data modality'
            self.labels = {0: "background", 1: "Label"}
            self.numTraining = 0
            self.numValidation = 0
            self.numTest = 0
            self.test = []
            self.training = []
            self.validation = []