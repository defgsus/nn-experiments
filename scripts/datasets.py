import math
import argparse
import random
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src.datasets import *
from src.util.image import *


def all_patch_dataset(
        shape: Tuple[int, int, int],
        shuffle: int = 200_000,
        file_shuffle: bool = True,
):
    ds = InterleaveIterableDataset(
        (
            kali_patch_dataset(shape, file_shuffle=file_shuffle),
            photo_patch_dataset(shape, "~/Pictures/photos/", file_shuffle=file_shuffle),
            photo_patch_dataset(shape, "~/Pictures/__diverse/", file_shuffle=file_shuffle),
        )
    )
    if shuffle:
        ds = IterableShuffle(ds, max_shuffle=shuffle)

    return ds


# currently about 3.8M patches
def kali_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path] = Path(__file__).resolve().parent.parent / "db/images/kali",
        file_shuffle: bool = True,
):
    return make_image_patch_dataset(
        #verbose_image=True,
        path=path,

        recursive=True,
        shape=shape,
        #max_images=1,
        max_image_bytes=1024 * 1024 * 1024 * 1,
        scales=partial(_scales_from_image_shape, shape, [2., 1., 1./2., 1./5, 1./10, 1./20., 1./30.]),
        stride=5,#_stride,
        interleave_images=20,
        #image_shuffle=5,
        patch_shuffle=10_000,
        file_shuffle=file_shuffle,
    )


def photo_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path] = Path("~/Pictures/photos").expanduser(),
        recursive: bool = True,
        file_shuffle: bool = True,
):
    def _stride(i_shape: Tuple[int, int]):
        # print(shape)
        size = min(i_shape)
        if size <= 512:
            return 5
        return shape[-2:]

    return make_image_patch_dataset(
        path=path, recursive=recursive,
        shape=shape,
        scales=partial(_scales_from_image_shape, shape, [1./12., 1./6, 1./3, 1.]),
        stride=_stride,
        interleave_images=20,
        file_shuffle=file_shuffle,
        transforms=[lambda x: x.clamp(0, 1)],
    )


def _scales_from_image_shape(
        shape: Tuple[int, ...],
        scales: Tuple[float, ...],
        image_shape: Tuple[int, int],
):
    size = min(image_shape)
    shape_size = min(shape[-2:])
    scale_list = []
    for s in scales:
        if s * size >= shape_size and s * size < 10_000:
            scale_list.append(s)
    return scale_list


# 20,103 seconds
AUDIO_FILENAMES_1 = ['~/Music/Alejandro Jodorowsky - [1971] El Topo OST (vinyl rip)/side1.mp3', '~/Music/Aphex Twin/Acoustica Alarm Will Sound Performs Aphex Twin/01 Cock_Ver 10.mp3', '~/Music/BR80-backup/ROLAND/LIVEREC/LIVE0000.WAV', '~/Music/BR80-backup/ROLAND/MASTERING/42FUNK.WAV', '~/Music/Bertolt Brecht & Kurt Weill - [1954] The Threepenny Opera (original off-broadway cast)/01 - prologue (spoken) gerald price.mp3', '~/Music/COIL - Absinthe/COIL - animal are you.mp3', '~/Music/COIL - Black Antlers/01-the gimp-sometimes.mp3', "~/Music/COIL - live at All Tomorrow's Parties, April 4, 2003...and the Ambulance Died in His Arms/01 - Triple Sun Introduction.mp3", "~/Music/Coil - [1991] Love's Secret Domain/01 - Disco Hospital.mp3", '~/Music/Crosby Stills  Nash & Young/carry on/Crosby Stills  Nash & Young - after the dolphin.mp3', '~/Music/Felix Kubin - Jane B. ertrinkt mit den Pferden/01 Wagner 99.mp3', "~/Music/Hitchhiker's Guide - Radio Play/Hitchhiker'sGuideEpisode-02,mp3", '~/Music/King Crimson/Discipline [30th Anniversary Edition] [Bonus Track]/01 Elephant Talk.mp3', "~/Music/King Crimson/Larks' Tongues in Aspic/01 Larks' Tongues in Aspic, Pt. 1.mp3", '~/Music/King Crimson/Three of a Perfect Pair- 30th Anniversary [Bonus Tracks]/01 Three of a Perfect Pair.mp3', '~/Music/King Crimson/Vrooom Vrooom Disc 1/01 Vrooom Vrooom.mp3', '~/Music/King Crimson/Vrooom Vrooom Disc 2/01 Conundrum.mp3', '~/Music/MODULE/42_modus.wav', '~/Music/MODULE/ATOMIK/INTRO.WAV', '~/Music/MODULE/FILMTHEA/sample.wav', '~/Music/MODULE/PATTERN/hoast84.wav', '~/Music/MODULE/TRIBB/02- MultiDrum_mp3.wav', '~/Music/MODULE/for_gonz/ATOMIK/INTRO.WAV', '~/Music/MODULE/sendung/UnitedSchneegl.wav', '~/Music/MODULE/werner/dolby/recycle samples/pianoarpeggio.wav', '~/Music/Primus/Primus - Antipop (Complete CD)(AlbumWrap)_ALBW.mp3', '~/Music/Ray Kurzweil The Age of Spiritual Machines/(audiobook) Ray Kurzweil - The Age of Spiritual Machines - 1 of 4.mp3', '~/Music/Scirocco/01 Zug.mp3', '~/Music/Symphony X/01 - Symphony X - The Damnation Game.mp3', '~/Music/VOLCANO THE BEAR - Amidst The Noise And Twigs (2007)/01 - Volcano The Bear - The Sting Of Haste.mp3', '~/Music/VOLCANO THE BEAR - Classic Erasmus Fusion (2006)/A01. Classic Erasmus Fusion.mp3', '~/Music/VOLCANO THE BEAR - Guess the Birds (2002)/01 - urchins at the harp.mp3', '~/Music/VOLCANO THE BEAR - Xvol/01. moon chorus.mp3', '~/Music/Volcano the bear - [2001] Five Hundred Boy Piano/01. Hairy Queen.mp3', '~/Music/Ys/01 emily.mp3', '~/Music/anke/DR-100_0809-mucke-tanja-traum-ist-aus.wav', '~/Music/diffusion/known-unknowns-02.wav', '~/Music/francis/scarlatti-k119.wav', '~/Music/grafft/Lotte/210429_1859.mp3', '~/Music/grafft/MUSIC/20200505_Bauchentscheidung.mp3', '~/Music/olli/24.07.19 eberhardt finaaaaaal.wav', '~/Music/record/20220624_untitled.wav', '~/Music/the who/Tommy/the who - 1921.mp3', '~/Music/theDropper/01 CD Track 01.mp3', '~/Music/yaggediyo.mp3']
AUDIO_FILENAMES_2 = ['~/Music/Alejandro Jodorowsky - [1971] El Topo OST (vinyl rip)/side2.mp3', '~/Music/Aphex Twin/Acoustica Alarm Will Sound Performs Aphex Twin/02 Logon Rock Witch.mp3', '~/Music/BR80-backup/ROLAND/LIVEREC/LIVE0001.WAV', '~/Music/BR80-backup/ROLAND/MASTERING/44BOND.WAV', '~/Music/Bertolt Brecht & Kurt Weill - [1954] The Threepenny Opera (original off-broadway cast)/02 - overture.mp3', '~/Music/COIL - Black Antlers/02-sex with sun ra (part 1 - saturnalia).mp3', "~/Music/COIL - live at All Tomorrow's Parties, April 4, 2003...and the Ambulance Died in His Arms/02 - Snow Falls Into Military Temples.mp3", "~/Music/Coil - [1991] Love's Secret Domain/02 - Teenage Lightning 1.mp3", '~/Music/Crosby Stills  Nash & Young/carry on/Crosby Stills  Nash & Young - almost cut my hair.mp3', '~/Music/Felix Kubin - Jane B. ertrinkt mit den Pferden/02 Vater Muss Die Stube Peitschen.mp3', "~/Music/Hitchhiker's Guide - Radio Play/Hitchhiker'sGuideEpisode-03.mp3", '~/Music/King Crimson/Discipline [30th Anniversary Edition] [Bonus Track]/02 Frame by Frame.mp3', "~/Music/King Crimson/Larks' Tongues in Aspic/02 Book of Saturday.mp3", '~/Music/King Crimson/Three of a Perfect Pair- 30th Anniversary [Bonus Tracks]/02 Modelk Man.mp3', '~/Music/King Crimson/Vrooom Vrooom Disc 1/02 Coda- Marine 475.mp3', '~/Music/King Crimson/Vrooom Vrooom Disc 2/02 Thela Hun Ginjeet.mp3', '~/Music/MODULE/43_monkeys have reached___.wav', '~/Music/MODULE/TRIBB/03- Unbenannt003_wav.wav', '~/Music/MODULE/sendung/buchstab01.wav', '~/Music/Ray Kurzweil The Age of Spiritual Machines/(audiobook) Ray Kurzweil - The Age of Spiritual Machines - 2 of 4.mp3', '~/Music/Scirocco/02 Nini Toscanè.mp3', '~/Music/Symphony X/02 - Symphony X - Dressed To Kill.mp3', '~/Music/VOLCANO THE BEAR - Amidst The Noise And Twigs (2007)/02 - Volcano The Bear - Before We Came To This Religion.mp3', '~/Music/VOLCANO THE BEAR - Classic Erasmus Fusion (2006)/A02. Did You Ever Feel Like Jesus¿.mp3', '~/Music/VOLCANO THE BEAR - Guess the Birds (2002)/02 - maureen memorium.mp3', '~/Music/VOLCANO THE BEAR - Xvol/02. snang dushko.mp3', '~/Music/Volcano the bear - [2001] Five Hundred Boy Piano/02. Seeker.mp3', '~/Music/Ys/02 monkey & bear.mp3', '~/Music/anke/DR-100_0809-mucke-tanja.wav', '~/Music/diffusion/known-unknowns-03.wav', '~/Music/francis/urdance_gsm_movt1.wav', '~/Music/grafft/Lotte/210429_1959.mp3', '~/Music/grafft/MUSIC/20200505_Eingecremt.mp3', '~/Music/olli/Du Schweigst_REV2_=FSM=__44.1-24.wav', '~/Music/the who/Tommy/the who - Amazing journey.mp3', '~/Music/theDropper/02 CD Track 02.mp3']


def audio_slice_dataset(
        path: Union[None, str, Path, Iterable[Union[str, Path]]] = None,
        recursive: bool = False,
        sample_rate: int = 44100,
        slice_size: int = 44100,
        stride: Optional[int] = None,
        interleave_files: Optional[int] = None,
        shuffle_files: bool = False,
        shuffle_slices: Optional[int] = None,
        mono: bool = False,
        seek_offset: float = 0.,
        spectral_shape: Optional[Tuple[int, int]] = None,
        spectral_patch_shape: Optional[Tuple[int, int]] = None,
        spectral_patch_stride: Optional[Tuple[int, int]] = None,
        spectral_normalize: Optional[int] = None,
        with_filename: bool = False,
        with_position: bool = False,
):
    if interleave_files is None:
        interleave_files = 1
    assert interleave_files > 0, interleave_files

    if path is None:
        path = AUDIO_FILENAMES_1

    ds = AudioSliceIterableDataset(
        path=path, recursive=recursive,
        sample_rate=sample_rate,
        slice_size=slice_size,
        stride=stride,
        interleave_files=interleave_files,
        shuffle_files=shuffle_files,
        mono=mono,
        seek_offset=seek_offset,
        with_filename=with_filename,
        with_position=with_position,
        #verbose=True,
    )
    if shuffle_slices:
        ds = IterableShuffle(ds, max_shuffle=shuffle_slices)

    if spectral_shape is not None:
        ds = TransformIterableDataset(
            ds,
            transforms=[
                AT.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=1024 * 2,
                    win_length=sample_rate // 30,
                    hop_length=sample_rate // spectral_shape[-1],
                    n_mels=spectral_shape[-2],
                    power=1.,
                ),
                lambda x: x[:, :, :spectral_shape[-1]]
            ],
        )

        if spectral_normalize:
            ds = NormalizeMaxIterableDataset(ds, spectral_normalize, clamp=(0, 1))

        if spectral_patch_shape is not None:
            ds = ImagePatchIterableDataset(
                ds,
                shape=spectral_patch_shape,
                stride=spectral_patch_stride,
                interleave_images=interleave_files,
            )

        # shape to [H, W]
        ds = TransformIterableDataset(
            ds,
            transforms=[
                lambda x: x.squeeze(0),
            ]
        )

    return ds


class RpgTileIterableDataset(IterableDataset):

    def __init__(self, shape: Tuple[int, int, int] = (3, 32, 32)):
        self.shape = shape

    def __iter__(self):
        yield from self._iter_tiles("~/prog/data/game-art/Castle2.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/PathAndObjects.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/mininicular.png", (8, 8))
        yield from self._iter_tiles("~/prog/data/game-art/items.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/roguelikeitems.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/apocalypse.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/tileset_1bit.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/MeteorRepository1Icons_fixed.png", (16, 16), (8, 0), (17, 17))
        yield from self._iter_tiles("~/prog/data/game-art/DENZI_CC0_32x32_tileset.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/overworld_tileset_grass.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/goodly-2x.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/Fruit.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/roguelikecreatures.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/metroid-like.png", (16, 16), limit=(128, 1000))
        yield from self._iter_tiles("~/prog/data/game-art/tilesheet_complete.png", (64, 64))
        yield from self._iter_tiles("~/prog/data/game-art/tiles-map.png", (16, 16))
        yield from self._iter_tiles("~/prog/data/game-art/base_out_atlas.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/build_atlas.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/obj_misk_atlas.png", (32, 32))
        yield from self._iter_tiles("~/prog/data/game-art/Tile-set - Toen's Medieval Strategy (16x16) - v.1.0.png", (16, 16))

    def _iter_tiles(self, name: str, shape: Tuple[int, int], offset: Tuple[int, int] = None, stride=None, limit=None):
        image = VF.to_tensor(PIL.Image.open(Path(name).expanduser()))

        if image.shape[0] != self.shape[0]:
            image = set_image_channels(image[:3], self.shape[0])

        if limit:
            image = image[..., :limit[0], :limit[1]]
        if offset:
            image = image[..., offset[0]:, offset[1]:]

        for patch in iter_image_patches(image, shape, stride=stride):
            if patch.std(1).mean() > 0.:
                #print(patch.std(1).mean())
                patch = VF.resize(patch, self.shape[-2:], VF.InterpolationMode.NEAREST, antialias=False)
                yield set_image_channels(patch, self.shape[0])
