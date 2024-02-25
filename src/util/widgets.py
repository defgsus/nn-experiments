import io

import torch
import torchvision.transforms.functional as VF

import ipywidgets
import PIL.Image


class ImageWidget(ipywidgets.Image):

    def set_pil(self, image: PIL.Image.Image):
        fp = io.BytesIO()
        image.save(fp, "png")
        fp.seek(0)
        self.format = "png"
        self.value = fp.read()

    def set_torch(self, image: torch.Tensor):
        image = VF.to_pil_image(image)
        self.set_pil(image)

