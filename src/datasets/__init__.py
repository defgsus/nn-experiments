from .audio_slice import AudioSliceIterableDataset
from .audio_spec import AudioSpecIterableDataset
from .aug import (
    CombineImageAugmentIterableDataset,
    ImageAugmentClassDataset,
    ImageNoiseDataset,
)
from .base_dataset import BaseDataset, WrapDataset
from .base_iterable import BaseIterableDataset
from .ca import TotalCADataset
from .classlogits import ClassLogitsDataset
from .contrastive import ContrastiveIterableDataset
from .contrastive_image import ContrastiveImageDataset
from .echords import EChordsIterableDataset
from .fefe import FefePostIterableDataset
from .filenames import FilenameDataset
from .filter import (
    FilterIterableDataset,
)
from .freeze import FreezeDataset
from .generative import (
    BoulderDashIterableDataset,
    IFSDataset, IFSClassIterableDataset,
    Kali2dDataset, Kali2dFilteredIterableDataset,
    MengerSponge2dDataset,
)
from .image_augmentation import ImageAugmentation
from .image_combine import ImageCombinePatchIterableDataset
from .image_dissimilar import DissimilarImageIterableDataset
from .image_encode import ImageEncodeIterableDataset
from .image_filter import ImageFilterIterableDataset
from .image_folder import ImageFolderIterableDataset
from .image_patch import (
    ImagePatchDataset, ImagePatchIterableDataset, make_image_patch_dataset, RandomImagePatchIterableDataset
)
from .image_scale import ImageScaleIterableDataset
from .image_source_target import ImageSourceTargetDataset, ImageSourceTargetCropDataset
from .interleave import InterleaveDataset, InterleaveIterableDataset
from .limit import (
    LimitDataset, SkipDataset, LimitIterableDataset, SkipIterableDataset,
    RepeatDataset, RepeatIterableDataset,
)
from .normalize import NormalizeMaxIterableDataset
from .randomcropall import RandomCropAllDataset
from .shuffle import ShuffleDataset, IterableShuffle
from .split import SplitIterableDataset
from .text import (
    make_compact_whitespace,
    TextSegmentIterableDataset, FileTextSegmentIterableDataset,
    TextGithubEventIterableDataset,
    TextWiki9SegmentIterableDataset,
)
from .textstride import TextStrideIterableDataset
from .unsplash import UnsplashDataset
from .pixelart import RpgTileIterableDataset
from .text2pix import TextToPixelIterableDataset
from .transform import TransformDataset, TransformIterableDataset
