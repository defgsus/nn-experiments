import torch

from .base import *
from src.algo.wangtiles import *
from .spatial_trans import SpatialTransformBase


class RandomSpatialTransformBase(SpatialTransformBase):
    IS_RANDOM = True


class RandomCrop(RandomSpatialTransformBase):
    """
    Crop at a random location.
    """
    NAME = "random_crop"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "size",
            "type": "int",
            "default": 224,
        },
        {
            "name": "pad_if_needed",
            "type": "bool",
            "default": True,
        },
        {
            "name": "padding_mode",
            "type": "select",
            "choices": ["constant", "edge", "reflect", "symmetric"],
            "default": "constant",
        }
    ]

    def __init__(
            self,
            size: int,
            pad_if_needed: bool,
            padding_mode: str,
    ):
        super().__init__()
        self.transform = VT.RandomCrop(size=size, pad_if_needed=pad_if_needed, padding_mode=padding_mode)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class RandomScale(SpatialTransformBase):
    """
    Randomly scale between fixed ratios
    """
    NAME = "random_scale"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "scale_min_xy",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
            "max": [2.**16, 2.**16],
        },
        {
            "name": "scale_max_xy",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
            "max": [2.**16, 2.**16],
        },
        {
            "name": "scale_xy_separately",
            "type": "bool",
            "default": False,
        },
        INTERPOLATION_PARAMETER,
    ]

    def __init__(
            self,
            scale_min_xy: Tuple[float, float],
            scale_max_xy: Tuple[float, float],
            scale_xy_separately: bool,
            interpolation: str,
    ):
        super().__init__()
        self.scale_min = scale_min_xy
        self.scale_max = scale_max_xy
        self.scale_xy_separately = scale_xy_separately
        self.interpolation = to_torch_interpolation(interpolation)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.scale_xy_separately:
            scale_x = random.uniform(self.scale_min[0], self.scale_max[0])
            scale_y = random.uniform(self.scale_min[1], self.scale_max[1])
        else:
            r = random.uniform(0, 1)
            scale_x = self.scale_min[0] + r * (self.scale_max[0] - self.scale_min[0])
            scale_y = self.scale_min[1] + r * (self.scale_max[1] - self.scale_min[1])

        new_shape = [
            max(1, int(scale_y * image.shape[-2])),
            max(1, int(scale_x * image.shape[-1]))
        ]
        if torch.Size(new_shape) == image.shape[-2:]:
            return image

        return VF.resize(
            image,
            size=new_shape,
            interpolation=self.interpolation,
            antialias=self.interpolation != VF.InterpolationMode.NEAREST,
        )


class RandomAffine(RandomSpatialTransformBase):
    """
    torchvision.transforms.RandomAffine
    """
    NAME = "random_affine"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "degrees_min_max",
            "type": "float2",
            "default": [-5., 5.],
            "step": [5., 5.],
            "min": [-360., -360],
            "max": [360., 360.],
        },
        {
            "name": "translate_xy",
            "type": "float2",
            "default": [0., 0.],
            "min": [-2., -2.],
            "max": [2., 2.],
        },
        {
            "name": "scale_min_max",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
            "max": [100., 100.],
        },
        {
            "name": "shear_min_max",
            "type": "float2",
            "default": [0., 0.],
            "step": [5., 5.],
            "min": [-360., -360.],
            "max": [360., 360.],
        },
        INTERPOLATION_PARAMETER
    ]

    def __init__(
            self,
            degrees_min_max: Tuple[float, float],
            translate_xy: Tuple[float, float],
            scale_min_max: Tuple[float, float],
            shear_min_max: Tuple[float, float],
            interpolation: str,
    ):
        super().__init__()
        self.transform = VT.RandomAffine(
            degrees=degrees_min_max,
            translate=None if translate_xy == (0, 0) else translate_xy,
            scale=None if scale_min_max == (1, 1) else scale_min_max,
            shear=None if shear_min_max == (0, 0) else (*shear_min_max, *shear_min_max),
            interpolation=to_torch_interpolation(interpolation),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)



class RandomTilingMap(RandomSpatialTransformBase):
    """
    Uses the defined tiling and renders a random tiled map
    """
    NAME = "random_tile_map"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "map_size",
            "type": "int2",
            "default": [4, 4],
            "min": [1, 1],
            "max": [1024, 1024],
        },
        {
            "name": "overlap",
            "type": "int2",
            "default": [0, 0],
            "min": [0, 0],
            "max": [1024, 1024],
        },
        {
            "name": "probability",
            "type": "float",
            "default": 1.,
            "min": 0.,
            "max": 1.,
        },
    ]

    def __init__(
            self,
            map_size: Tuple[int, int],
            overlap: Union[int, Tuple[int, int]],
            probability: float,
    ):
        super().__init__()
        self.map_size = map_size
        self.overlap = overlap
        self.probability = probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        def _render(template):
            if self.probability < 1.:
                if random.random() > self.probability:
                    return template

            tiling = self._clipig.input_tiling()
            if not tiling:
                return template

            tile_map = tiling.create_map_stochastic_scanline(self.map_size)

            return tiling.render_tile_map(template, tile_map, overlap=self.overlap)

        if image.ndim == 3:
            return _render(image)

        elif image.ndim == 4:
            return torch.concat([
                _render(i).unsqueeze(0)
                for i in image
            ])
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {image.shape}")


class RandomWangMap(RandomSpatialTransformBase):
    """
    Treats input as wang tile template and renders a random wang map
    """
    NAME = "random_wang_map"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "type",
            "type": "select",
            "default": "edge",
            "choices": ["edge", "corner"],
        },
        {
            "name": "num_colors",
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 1024,
        },
        {
            "name": "map_size",
            "type": "int2",
            "default": [4, 4],
            "min": [1, 1],
            "max": [1024, 1024],
        },
        {
            "name": "overlap",
            "type": "int2",
            "default": [0, 0],
            "min": [0, 0],
            "max": [1024, 1024],
        },
        {
            "name": "probability",
            "type": "float",
            "default": 1.,
            "min": 0.,
            "max": 1.,
        },
    ]

    def __init__(
            self,
            map_size: Tuple[int, int],
            num_colors: int,
            type: str,
            overlap: Union[int, Tuple[int, int]],
            probability: float,
    ):
        super().__init__()
        self.map_size = map_size
        self.overlap = overlap
        self.probability = probability
        self.wangtiles = WangTiles(get_wang_tile_colors(num_colors), mode=type)
        self.template: WangTemplate = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        def _render(template):
            if self.probability < 1.:
                if random.random() > self.probability:
                    return template

            if self.template is None or self.template.image.shape != template:
                self.template = self.wangtiles.create_template(template.shape)
            self.template.image = template

            return render_wang_map(
                self.template,
                wang_map_stochastic_scanline(self.wangtiles, self.map_size),
                overlap=self.overlap,
            ).to(image)

        if image.ndim == 3:
            return _render(image)

        elif image.ndim == 4:
            return torch.concat([
                _render(i).unsqueeze(0)
                for i in image
            ])
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {image.shape}")
