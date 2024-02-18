from .classic import (
    mnist_dataset,
    fmnist_dataset,
)
from .image_patch import (
    image_patch_dataset,
    all_image_patch_dataset,
)
from .noise import (
    noise_dataset
)
from .rpg import (
    RpgTileIterableDataset,
    rpg_tile_dataset,
    rpg_tile_dataset_3x32x32,
)
from .teletext import (
    TeletextIterableDataset,
    TeletextMatrixIterableDataset,
)
