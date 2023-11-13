from .audio_slice import AudioSliceIterableDataset
from .audio_spec import AudioSpecIterableDataset
from .ca import TotalCADataset
from .classfeatures import ClassFeaturesDataset
from .contrastive import ContrastiveIterableDataset
from .contrastive_image import ContrastiveImageDataset
from .image_augmentation import ImageAugmentation
from .image_dissimilar import DissimilarImageIterableDataset
from .image_encode import ImageEncodeIterableDataset
from .image_filter import ImageFilterIterableDataset
from .image_folder import ImageFolderIterableDataset
from .image_patch import ImagePatchIterableDataset, make_image_patch_dataset
from .image_scale import ImageScaleIterableDataset
from .interleave import InterleaveIterableDataset
from .limit import LimitIterableDataset
from .generative import (
    Kali2dDataset, Kali2dFilteredIterableDataset
)
from .normalize import NormalizeMaxIterableDataset
from .shuffle import IterableShuffle
from .split import SplitIterableDataset
from .transform import TransformDataset, TransformIterableDataset
