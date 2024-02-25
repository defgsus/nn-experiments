from pathlib import Path

from torch.utils.data import DataLoader, IterableDataset
#from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
import torch.utils.data

from tqdm import tqdm

from src.datasets import *


class PassThrough(IterableDataset):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        items = list(self.source)
        print("GOT", len(items))
        for item in items:
            yield item
            #yield item


def main():
    ds = ImageFolderIterableDataset(
        #folder=Path("~/Pictures/__diverse/").expanduser(),
        root=str(Path("~/Pictures/__diverse/").expanduser()),
    )
    #ds = ImageAugmentation(ds, augmentations=[
    #    VT.RandomRotation(degrees=20),
    #    VT.RandomPerspective(),
    #])
    #ds = PassThrough(ds)
    dl = DataLoader(
        ds,
        #num_workers=8,
        #batch_size=2,
    )

    count = 0
    for i in tqdm(dl):
        #print(i.shape)
        count += 1
        #if count >= 5:
        #    break

    print("COUNT:", count)


def test_loader_logic():

    class RangeDataset(IterableDataset):
        def __init__(self, num: int):
            super().__init__()
            self.num = num

        def __iter__(self):
            numbers = list(range(self.num))

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                numbers = numbers[worker_info.id::worker_info.num_workers]

            print("YIELDING", numbers, worker_info)
            yield from numbers

    ds = RangeDataset(10)
    ds = PassThrough(ds)
    dl = DataLoader(ds, num_workers=4)

    result = list(dl)

    print("RESULT", len(result), result)


if __name__ == "__main__":
    main()
    #test_loader_logic()
