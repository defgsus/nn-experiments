from pathlib import Path

from torch.utils.data import IterableDataset

from src.util.files import iter_ndjson


class EChordsIterableDataset(IterableDataset):

    def __init__(self):
        self.filename = Path("~/prog/python/github/nn-experiments/datasets/echords.ndjson.gz").expanduser()
        assert self.filename.exists(), f"Did not find: {self.filename}"

    def __len__(self):
        return 372051 - 1

    def __iter__(self):
        for data in iter_ndjson(self.filename):
            for key, value in data.items():
                if value is None:
                    data[key] = ""

            if len(data["text"]) < 100:
                continue

            data["text"] = data["text"].replace("\r", "")

            yield data
