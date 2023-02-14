from lib import *

class Loader:
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