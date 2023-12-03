import time
from pathlib import Path
import glob
import os
import warnings
import datetime
from typing import Union, Generator, List, Optional, Iterable

import requests
from tqdm import tqdm

from src.util.files import iter_ndjson


class GHArchive:

    def __init__(
            self,
            raw_path: Union[str, Path] = Path("~/prog/data/gharchive/").expanduser(),
            verbose: bool = True,
    ):
        self.raw_path = Path(raw_path)
        self.verbose = verbose

    def iter_events(
            self,
            day: datetime.date,
            event_type: Optional[str] = None,
            hours: Union[None, int, Iterable[int]] = None,
            probability: float = 1.,
    ) -> Generator[dict, None, None]:
        id_set = set()

        filter = None
        if event_type:
            filter = lambda line: f'"type":"{event_type}"' in line

        if hours is None:
            hours = list(range(0, 24))
        elif isinstance(hours, int):
            hours = (hours, )
        else:
            hours = set(hours)

        for hour in hours:

            filename = self.get_event_file(day, hour)
            if not filename:
                continue

            for event in iter_ndjson(filename, filter=filter, probability=probability):

                # skip the old events
                if not event.get("id"):
                    warnings.warn("Old events (without id) are skipped a.t.m.")
                    continue

                if event["id"] not in id_set:
                    yield event
                    id_set.add(event["id"])

                    # remove the oldest IDs
                    if len(id_set) >= 1_000_000:
                        for id in sorted(id_set)[:500_000]:
                            id_set.remove(id)

    def iter_commits(
            self,
            day: datetime.date,
            hours: Union[None, int, Iterable[int]] = None,
    ) -> Generator[dict, None, None]:
        for event in self.iter_events(day=day, hours=hours, event_type="PushEvent"):
            for commit in event["payload"]["commits"]:
                yield commit

    def get_event_file(self, day: datetime.date, hour: int) -> Optional[Path]:
        filename = f"{day.year}-{day.month:02d}-{day.day:02d}-{hour}.json.gz"
        local_filename = self.raw_path / f"{day.year}" / filename
        if not local_filename.exists():
            if not self._download(filename, local_filename):
                return
        return local_filename

    def _download(self, filename: str, local_filename: Path, chunk_size: int = 64_000, num_tries: int = 5) -> bool:
        if self.verbose:
            print(f"Downloading https://data.gharchive.org/{filename}")

        for i in range(num_tries):
            try:
                return self._download_impl(filename, local_filename, chunk_size)
            except requests.ConnectionError as e:
                if i == num_tries - 1:
                    raise

                if self.verbose:
                    print(f"retrying after {type(e).__name__}: {e}")
                    time.sleep(.5)

    def _download_impl(self, filename: str, local_filename: Path, chunk_size: int = 64_000) -> bool:
        with requests.get(f"https://data.gharchive.org/{filename}", stream=True) as r:
            if r.status_code != 200:
                if self.verbose:
                    print(f"\n\nNOT FOUND {r.request.url}, got status {r.status_code}")
                return False

            size = int(r.headers['content-length'])

            os.makedirs(local_filename.parent, exist_ok=True)
            try:
                with tqdm(total=size, desc=f"downloading {filename}", disable=not self.verbose) as progress:
                    with open(str(local_filename), "wb") as fp:
                        for data in r.iter_content(chunk_size=chunk_size):
                            fp.write(data)
                            progress.update(len(data))
                return True

            except:
                # don't leave half files
                os.remove(str(local_filename))
                raise

    @staticmethod
    def _sort_key(k: str):
        idx = k.rfind(os.sep) + 1
        if k[idx+12] == ".":
            return k[:idx+11] + "0" + k[idx+11:]
        return k
