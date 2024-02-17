from .base import *
from src.algo.wangtiles import *


class SpatialTransformBase(TransformBase):
    CLASS = "spatial"


class Scale(SpatialTransformBase):
    """
    Scale by a fixed ratio
    """
    NAME = "scale"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "scale",
            "type": "float",
            "default": 2.,
            "min": 0,
        },
        INTERPOLATION_PARAMETER,
    ]

    def __init__(self, scale: float, interpolation: str):
        super().__init__()
        self.scale = scale
        self.interpolation = to_torch_interpolation(interpolation)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        new_shape = [max(1, int(s * self.scale)) for s in image.shape[-2:]]

        return VF.resize(
            image,
            size=new_shape,
            interpolation=self.interpolation,
            antialias=self.interpolation != VF.InterpolationMode.NEAREST,
        )


class RandomCrop(SpatialTransformBase):
    """
    Crop at a random location.
    """
    NAME = "random_crop"
    IS_RESIZE = True
    IS_RANDOM = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "size",
            "type": "int",
            "default": 224,
        },
    ]

    def __init__(self, size: int):
        super().__init__()
        self.transform = VT.RandomCrop(size=size, pad_if_needed=True)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class RandomAffine(SpatialTransformBase):
    """
    torchvision.transforms.RandomAffine
    """
    NAME = "random_affine"
    IS_RESIZE = True
    IS_RANDOM = True
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


class Repeat(SpatialTransformBase):
    """
    Repeat the image in x and y direction
    """
    NAME = "repeat"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "repeat_xy",
            "type": "int2",
            "default": [2, 2],
            "min": [1, 1],
            "max": [100, 100],
        },
    ]

    def __init__(self, repeat_xy: Tuple[int]):
        super().__init__()
        self.repeat_xy = repeat_xy

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.repeat(1, self.repeat_xy[1], self.repeat_xy[0])


class Padding(SpatialTransformBase):
    """
    Pad individual edges of the image
    """
    NAME = "padding"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "pad_left",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_top",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_right",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_bottom",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
    ]

    def __init__(
            self,
            pad_left: int,
            pad_top: int,
            pad_right: int,
            pad_bottom: int,
    ):
        super().__init__()
        self.pad_left = pad_left
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return VF.pad(image, [self.pad_left, self.pad_top, self.pad_right, self.pad_bottom])



class Blur(SpatialTransformBase):
    NAME = "blur"
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "kernel_size",
            "type": "int2",
            "default": [3, 3],
            "min": [1, 1],
        },
        {
            "name": "sigma",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
        },
        {
            "name": "mix",
            "type": "float",
            "default": 1.,
            "min": 0.,
            "max": 1.,
        },
    ]

    def __init__(
            self,
            kernel_size: Tuple[int, int],
            sigma: Tuple[float, float],
            mix: float,
    ):
        super().__init__()
        self.sigma = sigma
        self.mix = mix
        self.kernel_size = [
            max(1, k+1 if k % 2 == 0 else k)
            for k in reversed(kernel_size)
        ]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        blurred = VF.gaussian_blur(image, self.kernel_size, self.sigma)
        if self.mix != 1.:
            blurred = image * (1. - self.mix) + self.mix * blurred
        return blurred


class RandomWangMap(SpatialTransformBase):
    """
    Treats input as wang tile template and renders a random wang map
    """
    NAME = "random_wang_map"
    IS_RESIZE = True
    IS_RANDOM = True
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
