import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable


from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src import console
from src.train.train_autoencoder import Trainer
from src.models.rbm import *
from src.datasets import ImageFolderIterableDataset, ImageAugmentation, IterableShuffle, TotalCADataset
from src.util.image import get_images_from_iterable


class FontDataset(IterableDataset):

    def __init__(self, shape=(3, 32, 32), num_samples: int = 10_000_000):
        self.shape = shape
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def check_image(self, image) -> bool:
        if torch.all(image.reshape(3, -1).std(1) < 0.1):
            return False
        return True

    def random_text(self) -> str:
        return "".join(
            chr(random.randint(32, 100))
            for i in range(random.randint(1, 10))
        )

    def __iter__(self):
        font_path = Path("~/.local/share/fonts").expanduser()
        fonts = list(font_path.glob("*ttf"))
        fonts = [ImageFont.truetype(str(f)) for f in fonts[:10]]
        count = 0
        while count < self.num_samples:
            image = PIL.Image.new(
                "RGB", (random.randint(8, self.shape[-1]), random.randint(8, self.shape[-2])),
                #color=(random.randrange(256), random.randrange(256), random.randrange(256)),
            )
            draw = ImageDraw.ImageDraw(image)
            for i in range(random.randint(1, random.randint(1, 3))):
                draw.text(
                    (random.randrange(image.width//10+1), random.randrange(image.height)), self.random_text(),
                    font=random.choice(fonts),
                    fill=(random.randrange(256), random.randrange(256), random.randrange(256)),
                )

            image = VF.to_tensor(image)
            if random.random() < .5:
                image = VT.RandomRotation(30)(image)
            image = VF.resize(image, self.shape[1:])

            if self.check_image(image):
                yield image
                count += 1


class RBMTrainer(Trainer):

    def write_step(self):
        shape = self.hparams["shape"]
        outputs = self.model.forward(self.validation_batch)
        self.log_image("validation_output", outputs.unsqueeze(0))

        org, recon = self.model.contrastive_divergence(self.validation_batch[:8])
        recon2 = self.model.gibbs_sample(self.validation_batch[:8], num_steps=20)
        self.log_image("reconstruction", make_grid(
            get_images_from_iterable(org.view(8, *shape), squeezed=True, num=8)
            + get_images_from_iterable(recon.view(8, *shape), squeezed=True, num=8)
            + get_images_from_iterable(recon2.view(8, *shape), squeezed=True, num=8),
            nrow=8
        ))

    def X_every_1000000(self):
        print("increasing model size")
        self.model.duplicate_output_cells()
        self.optimizers = [
            torch.optim.Adadelta(self.model.parameters(), lr=1.),
        ]


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (1, 32, 32)
    # ds = TensorDataset(torch.load("./datasets/diverse-64x64-aug4.pt"))
    #ds = TensorDataset(torch.load("./datasets/diverse-32x32-aug32.pt"))
    #ds = TensorDataset(torch.load("./datasets/diverse-32x32-std01.pt"))
    #ds = TensorDataset(torch.load("./datasets/fonts-regular-32x32.pt"))
    #ds = TensorDataset(torch.load("./datasets/photos-32x32-std01.pt"))
    ds = TotalCADataset(SHAPE[-2:], num_iterations=10, init_prob=.5, wrap=True, transforms=[lambda x: x.unsqueeze(0)])
    assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    #ds = ConcatDataset([ds, ds2])

    train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    model = RBM(math.prod(SHAPE), 32)
    print(model)

    trainer = RBMTrainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_inputs_between_validations=100_000,
        #data_loader=DataLoader(train_ds, shuffle=True, batch_size=10),
        data_loader=DataLoader(train_ds, batch_size=50, num_workers=5),
        validation_loader=DataLoader(test_ds),
        optimizers=[
            #torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0.001),
            torch.optim.Adadelta(model.parameters(), lr=1.),
        ],
        hparams={
            "shape": SHAPE,
        },
        weight_image_kwargs={
            "shape": SHAPE,
        }
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
