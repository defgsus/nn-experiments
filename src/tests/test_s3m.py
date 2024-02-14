import json
import unittest

from tqdm import tqdm

from src.util.s3m import *
from tests.base import TestBase


class TestS3m(TestBase):

    def test_100_load(self):
        s3m = S3m.from_file(self.DATA_PATH / "tracker/SOUNNGG.S3M")
        s3m.dump()
        # s3m.dump_pattern(s3m.patterns[0])
        #for inst in s3m.instruments:
        #    print(vars(inst))

    def test_200_render(self):
        s3m = S3m.from_file(self.DATA_PATH / "tracker/SOUNNGG.S3M")
        renderer = S3mRenderer(s3m)

        for i in range(3):
            renderer.process(1024)

        buffer_size = 1024
        num_samples = int(s3m.calc_length() * renderer.samplerate)
        with tqdm(desc="rendering", total=num_samples) as progress:
            c = 0
            while not renderer.endofsong:
                renderer.process(buffer_size)
                progress.update(buffer_size)
                c += 1
                if c > 10:
                    break
                #if c % 10 == 0:
                #    print(f"{renderer.position:3} {renderer.row:2} {renderer.tick}")

        renderer.reset()
        while not renderer.endofsong:
            flags = "".join(str(i) if renderer.flags >> i else "." for i in range(8, 0, -1))
            chan_info = "|".join(
                f"{ch.noteon:3} {ch.sample:3} {ch.volume:3} {ch.command:3}"
                for ch in renderer.channels[:4]
            )
            #print(f"{renderer.position:3} {renderer.row:2} {renderer.tick} {flags} : {chan_info}")
            renderer._process_tick()
