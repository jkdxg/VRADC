from .field import RawField, Merge, ImageDetectionsField, TextField, ImageField, VisualImageField
from .dataset import COCO
from torch.utils.data import DataLoader as TorchDataLoader
from .field_online import ImageField_online
from .dataset_online import COCO_online

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
