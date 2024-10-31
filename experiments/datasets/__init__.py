from .boulderdash import (
    boulderdash_dataset_32x32,
)
from .classic import (
    mnist_dataset,
    fmnist_dataset,
)
from .clip_noise import (
    ClipNoiseDataset,
    ClipNoiseMixDataset,
)
from .image_patch import (
    image_patch_dataset,
    all_image_patch_dataset,
)
from .kali import (
    kali_patch_dataset
)
from .noise import (
    noise_dataset
)
from .pixelart import PixelartDataset
from .teletext import (
    TeletextIterableDataset,
    TeletextPixelIterableDataset,
    TeletextMatrixIterableDataset,
)
