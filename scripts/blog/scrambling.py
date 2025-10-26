import os
import random
import hashlib
import copy
from pathlib import Path

from fontTools.ttLib import TTFont


class ScrambledFont:

    def __init__(
            self,
            output_path: Path,
            # TODO: adding any of the , ; : etc.. characters leads to an infinite recursion in font.save()
            chars_to_scramble: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.", #,;:!?",
            # the following characters are not usable anymore, but that's fine for my purposes
            # Note that they must exists in the original font file because i could not figure out
            #   how to ADD new glyphs to the font-file via fontTools
            extra_chars: str = "ϗϘϙϚϛϜϝϞϟϠϡϢϣϤϥϦϧϨϩϪϫϬϭϮϯϰϱϲϳϴϵ϶ϷϸϹϺϻϼϽϾϿЀЁЂЃЄЅІЇЈЉЊЋЌЍЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрсљњћќѝўџѠѡѢѣѤѥѦѧѨѩѪѫѬѭѮѯѰѱѲѳѴѵѶѷѸѹѺѻѼѽѾѿҀҁҊҋҌҍҎҏҐґҒғҔҕҖҗҘҙҚқҜҝҞҟҠҡҢңҤҥҦҧҨҩҪҫҬҭҮүҰұҲҳҴҵҶҷҸҹҺһҼҽҾҿӀӁӂӃӄӅӆӇӈӉӊӋӌӍӎӏӶӷӸӹӺӻӼӽӾӿԀԁԂԃԄԅԆԇԈԉԊԋԌԍԎԏԐԑԒԓԔԕԖԗԘԙԚԛԜԝԞԟԠԡԢԣԤԥԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՙ՚՛՜՝՞՟աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև",
            seed: int = 23,
            font_files: dict[str, str] = {
                "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                #"bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            },
    ):
        self.seed = seed
        self.chars_to_scramble = chars_to_scramble
        self.original_font_files = font_files
        self.extra_chars = extra_chars
        self.hash = hashlib.sha3_256(
            f"{self.seed}{self.chars_to_scramble}{self.extra_chars}{self.original_font_files}".encode()
        ).hexdigest()[:10]

        self.font_files = {
            key: output_path / f"scrambled-{key}-{self.hash}.woff2"
            for key in self.original_font_files
        }

        scrambled_chars = list(chars_to_scramble)
        rng = random.Random(self.seed)
        rng.shuffle(scrambled_chars)

        self.scramble_map = {
            ch1: [ch2]
            for ch1, ch2 in zip(self.chars_to_scramble, scrambled_chars)
        }
        extra_chars = list(self.extra_chars)
        rng.shuffle(extra_chars)
        for i, c in enumerate(extra_chars):
            self.scramble_map[self.chars_to_scramble[i % len(self.chars_to_scramble)]].append(c)
        #import json
        #print(json.dumps(self.scramble_map, indent=2))
        #print(self.scramble_map)

    def render_font_files(self):
        for key, input_filename in self.original_font_files.items():
            output_file = self.font_files[key]
            if not output_file.exists():
                self.render_font_file(Path(input_filename), output_file)

    def render_font_file(self, input_filename: Path, output_filename: Path):
        print(f"Rendering {output_filename}")
        font = TTFont(input_filename)
        font.ensureDecompiled()

        ord_to_name = font.getBestCmap()
        scramble_map = {
            ord_to_name[ord(ch1)]: [ord_to_name[ord(c)] for c in ch2]
            for ch1, ch2 in self.scramble_map.items()
        }

        gs = font.getGlyphSet()
        original_glyphs = {}
        for ch1 in scramble_map.keys():
            original_glyphs[ch1] = (
                copy.deepcopy(font.tables["glyf"][ch1]),
                copy.deepcopy(gs.hMetrics[ch1]),
            )

        for ch1, ch2 in scramble_map.items():
            for c in ch2:
                font.tables["glyf"][c], gs.hMetrics[c] = original_glyphs[ch1]

        os.makedirs(output_filename.parent, exist_ok=True)
        font.save(output_filename)
