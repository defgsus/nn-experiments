"""
Bootstrapping a pixel-art dataset

1. collect links to opengameart.org pages and store in `OPEN_GAME_ART_URLS`
2. call `download_all()` (`python src/dataset/bootstrap.py download`) which downloads all the pages
   with all their content and extracts the zip files if present. just for easier browsing on local disk.
3. define each file to use in `TILE_CONFIGS`. This one is the hard part, finding correct tile size,
   offset, limit, ignore certain tiles with copyright letters aso... This can best be done in a `jupyter notebook`
   using the `RpgTileBootstrapIterableDataset` from this file.
4. Run `python src/dataset/bootstrap.py filter` to create the `tile-config.ndjson` file. This will filter out
   equal tiles.

"""
import argparse
import fnmatch
import json
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Union

import torch
from torch.utils.data import IterableDataset
import torchvision.transforms.functional as VF
import PIL.Image
from tqdm import tqdm

from src.util.image import iter_image_patches, set_image_channels
from src.datasets import DissimilarImageIterableDataset


STORAGE_DIRECTORY = "~/prog/data/game-art/DOWN/"

OPEN_GAME_ART_URLS = [
    "https://opengameart.org/content/1-bit-doomgeon-kit",
    "https://opengameart.org/content/lots-of-free-2d-tiles-and-sprites-by-hyptosis",
    "https://opengameart.org/content/zelda-like-tilesets-and-sprites",
    "https://opengameart.org/content/dungeon-crawl-32x32-tiles",
    "https://opengameart.org/content/whispers-of-avalon-grassland-tileset",
    "https://opengameart.org/content/dungeon-tileset",
    "https://opengameart.org/content/a-blocky-dungeon",
    "https://opengameart.org/content/tiny-16-basic",
    "https://opengameart.org/content/lpc-tile-atlas",
    "https://opengameart.org/content/the-field-of-the-floating-islands",
    "https://opengameart.org/content/dawnlike-16x16-universal-rogue-like-tileset-v181",
    "https://opengameart.org/content/platformer-art-complete-pack-often-updated",
    "https://opengameart.org/content/castle-tiles-for-rpgs",
    "https://opengameart.org/content/rpg-tiles-cobble-stone-paths-town-objects",
    "https://opengameart.org/content/32x32-fantasy-tileset",
    "https://opengameart.org/content/generic-platformer-tiles",
    "https://opengameart.org/content/lpc-tile-atlas2",
    "https://opengameart.org/content/grassland-tileset",
    "https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental",
    "https://opengameart.org/content/16x16-fantasy-tileset",
    "https://opengameart.org/content/2d-lost-garden-zelda-style-tiles-resized-to-32x32-with-additions",
    "https://opengameart.org/content/tiled-terrains",
    "https://opengameart.org/content/worldmapoverworld-tileset",
    "https://opengameart.org/content/dungeon-tileset-1",
    "https://opengameart.org/content/cave-tileset-0",
    "https://opengameart.org/content/magic-cliffs-environment",
    "https://opengameart.org/content/seasonal-platformer-tiles",
    "https://opengameart.org/content/top-down-sci-fi-shooter-pack",
    "https://opengameart.org/content/browserquest-sprites-and-tiles",
    "https://opengameart.org/content/sci-fi-platform-tiles",
    "https://opengameart.org/content/orthographic-outdoor-tiles",
    "https://opengameart.org/content/simple-broad-purpose-tileset",
    "https://opengameart.org/content/minimalist-pixel-tileset",
    "https://opengameart.org/content/lpc-farming-tilesets-magic-animations-and-ui-elements",
    "https://opengameart.org/content/country-side-platform-tiles",
    "https://opengameart.org/content/desert-tileset-0",
    "https://opengameart.org/content/outdoor-tiles-again",
    "https://opengameart.org/content/isometric-64x64-medieval-building-tileset",
    "https://opengameart.org/content/rpg-indoor-tileset-expansion-1",
    "https://opengameart.org/content/old-frogatto-tile-art",
    "https://opengameart.org/content/platform-pixel-art-assets",
    "https://opengameart.org/content/lpc-dungeon-elements",
    "https://opengameart.org/content/classical-ruin-tiles",
    "https://opengameart.org/content/top-down-dungeon-tileset",
    "https://opengameart.org/content/16x16-game-assets",
    "https://opengameart.org/content/rpg-town-pixel-art-assets",
    "https://opengameart.org/content/cave-tileset",
    "https://opengameart.org/content/zoria-tileset",
    "https://opengameart.org/content/medieval-building-tiles",
    "https://opengameart.org/content/gothicvania-town",
    "https://opengameart.org/content/abandonauts-8x8-tile-assets",
    "https://opengameart.org/content/lpc-terrain-repack",
    "https://opengameart.org/content/tuxemon-tileset",
    "https://opengameart.org/content/post-apocalyptic-16x16-tileset-update1",
    "https://opengameart.org/content/isometric-miniature-dungeon",
    "https://opengameart.org/content/basic-map-32x32-by-ivan-voirol",
    "https://opengameart.org/content/slates-32x32px-orthogonal-tileset-by-ivan-voirol",
    "https://opengameart.org/content/platformer-art-pixel-redux",
    "https://opengameart.org/content/pixel-art-castle-tileset",
    "https://opengameart.org/content/denzis-public-domain-art",
    "https://opengameart.org/content/lpc-house-insides",
    "https://opengameart.org/content/free-desert-platformer-tileset",
    "https://opengameart.org/content/a-cute-dungeon",
    "https://opengameart.org/content/victims-and-villagers",
    "https://opengameart.org/content/simerion-tiles-and-images",
    "https://opengameart.org/content/metroid-like",
    "https://opengameart.org/content/kenney-16x16",
    "https://opengameart.org/content/isometric-road-tiles-nova",
    "https://opengameart.org/content/pixel-world-hex-tileset",
    "https://opengameart.org/content/prototyping-2d-pixelart-tilesets",
    "https://opengameart.org/content/16x16-snowy-town-tiles",
    "https://opengameart.org/content/free-platformer-game-tileset",
    "https://opengameart.org/content/super-seasonal-platformer-tiles",
    "https://opengameart.org/content/64x64-isometric-roguelike-tiles",
    "https://opengameart.org/content/instant-dungeon-v14-art-pack",
    "https://opengameart.org/content/lpc-terrain-extension",
    "https://opengameart.org/content/grotto-escape-ii-environment",
    "https://opengameart.org/content/8x8-tileset-by-soundlust",
    "https://opengameart.org/content/platform-tileset-nature",
    "https://opengameart.org/content/lpc-medieval-village-decorations",
    "https://opengameart.org/content/grass-and-water-tiles",
    "https://opengameart.org/content/seven-kingdoms",
    "https://opengameart.org/content/lpc-animated-water-and-fire",
    "https://opengameart.org/content/dawnblocker-ortho",
    "https://opengameart.org/content/lpc-terrains",
    "https://opengameart.org/content/isometric-64x64-outside-tileset",
    "https://opengameart.org/content/platformer-art-extended-tilesets",
    # ^ page 4 of opengameart.org favorite "tile" 2d art
    "https://opengameart.org/content/winter-platformer-game-tileset",
    "https://opengameart.org/content/lpc-a-shootem-up-complete-graphic-kit",
    "https://opengameart.org/content/roguedb32",
    "https://opengameart.org/content/micro-tileset-overworld-and-dungeon",
    "https://opengameart.org/content/free-sci-fi-platformer-game-tileset",
    "https://opengameart.org/content/platformer-pack-redux-360-assets",
    "https://opengameart.org/content/hexagon-pack-310x",
    "https://opengameart.org/content/16x16-tiles",
    "https://opengameart.org/content/rpg-item-set",
    "https://opengameart.org/content/horror-tile-set",
    "https://opengameart.org/content/dutone-tileset-objects-and-character",
    "https://opengameart.org/content/lpc-wooden-furniture",
    "https://opengameart.org/content/lpc-submissions-merged",
    "https://opengameart.org/content/unknown-horizons-tileset",
    "https://opengameart.org/content/retro-tileset",
    "https://opengameart.org/content/isometric-city",
    "https://opengameart.org/content/updated-generic-platformer-tiles",
    "https://opengameart.org/content/space-war-man-platform-shmup-set",
    "https://opengameart.org/content/3x-updated-32x32-scifi-roguelike-enemies",
    "https://opengameart.org/content/mines-of-sharega",
    "https://opengameart.org/content/isometric-miniature-library",
    "https://opengameart.org/content/ground-tileset-grass-sand",
    "https://opengameart.org/content/pixelantasy",
    "https://opengameart.org/content/side-scrolling-fantasy-themed-game-assets",
    "https://opengameart.org/content/dark-forest-town-tile-set-0",
    "https://opengameart.org/content/gotthicvania-swamp",
    "https://opengameart.org/content/isaiah658s-pixel-pack-1",
    "https://opengameart.org/content/goblin-caves",
    "https://opengameart.org/content/2d-platformer-volcano-pack-11",
    "https://opengameart.org/content/steampunk-level-tileset-mega-pack-level-tileset-16x16",
    "https://opengameart.org/content/isometric-miniature-prototype",
    "https://opengameart.org/content/alien-planet-platformer-tileset",
    "https://opengameart.org/content/tiny-adventure-pack",
    "https://opengameart.org/content/pixvoxel-colorful-isometric-wargame-sprites",
    "https://opengameart.org/content/denzis-scifi-tilesets",
    "https://opengameart.org/content/lpc-misc-tile-atlas-interior-exterior-trees-bridges-furniture",
    "https://opengameart.org/content/stone-bridge-tiles-32x32",
    "https://opengameart.org/content/lpc-heroine",
    "https://opengameart.org/content/denzis-32x32-orthogonal-tilesets",
    "https://opengameart.org/content/bevouliin-free-game-obstacle-spikes",
    "https://opengameart.org/content/2d-platform-ground-stone-tiles",
    "https://opengameart.org/content/16-color-64-tiles-dungeon-01",
    "https://opengameart.org/content/space-merc-redux-platform-shmup-tileset",
    "https://opengameart.org/content/tileset-1bit-color-extention",
    "https://opengameart.org/content/trees-mega-pack-cc-by-30-0",
    "https://opengameart.org/content/8-colors-4-stages-1-layer-64-16x16-tiles",
    "https://opengameart.org/content/multi-platformer-tileset-grassland-old-version",
    "https://opengameart.org/content/pixel-platformer-tile-set",
    "https://opengameart.org/content/200-tileset-mega-metroidvania",
    "https://opengameart.org/content/space-shooter-sprites",
    "https://opengameart.org/content/colored-16x16-fantasy-tileset",
    "https://opengameart.org/content/old-frogatto-tiles-pt2",
    # ^ page 7
    "https://opengameart.org/content/puhzil",
    "https://opengameart.org/content/1-layer-8-bit-15-color-4-stages-tileset",
    "https://opengameart.org/content/sokoban-100-tiles",
    "https://opengameart.org/content/gameboy-rpg-tile",
    "https://opengameart.org/content/exterior-32x32-town-tileset",
    "https://opengameart.org/content/2d-platformer-jungle-pack",
    "https://opengameart.org/content/lpc-style-wood-bridges-and-steel-flooring",
    "https://opengameart.org/content/denzis-sidescroller-tilesets",
    "https://opengameart.org/content/4-color-dungeon-bricks-16x16",
    "https://opengameart.org/content/map-tile",
    "https://opengameart.org/content/desert-tileset",
    "https://opengameart.org/content/rpg-tiles-%E2%80%94-forest-meadows-outdoor",
    "https://opengameart.org/content/mini-roguelike-8x8-tiles",
    "https://opengameart.org/content/freeart-resuable-art-for-building-houses",
    "https://opengameart.org/content/early-80s-arcade-pixel-art-dungeonsslimes-walls-power-ups-etc",
    "https://opengameart.org/content/wyrmsun-cc0-over-900-items",
    "https://opengameart.org/content/modified-isometric-64x64-outside-tileset",
    "https://opengameart.org/content/animated-ocean-water-tile",
    "https://opengameart.org/content/denzis-16x16-orthogonal-tilesets",
    "https://opengameart.org/content/cobblestone-tileset",
    "https://opengameart.org/content/basic-32x32-sci-fi-tiles-for-roguelike",
    "https://opengameart.org/content/lpc-grave-markers-remix",
    "https://opengameart.org/content/minimal-sidescroller-tileset-expansion",
    "https://opengameart.org/content/lpc-alchemy",
    "https://opengameart.org/content/muckety-mudskipper-sprites-and-tiles",
    "https://opengameart.org/content/top-down-asset-pack-1-ctatz",
    "https://opengameart.org/content/city-pixel-tileset",
    "https://opengameart.org/content/castle-walls-isometric-64-x-128",
    "https://opengameart.org/content/rock-tileset",
    "https://opengameart.org/content/cute-sprites-pack-1",
    "https://opengameart.org/content/tiny16-tileset",
    "https://opengameart.org/content/rpg-sheet",
    "https://opengameart.org/content/roguelike-bosses",
    "https://opengameart.org/content/8-color-full-game-sprite-tiles",
    "https://opengameart.org/content/lpc-wooden-ship-tiles",
    "https://opengameart.org/content/instant-dungeon-art-pack",
    "https://opengameart.org/content/2d-platformer-desert-pack",
    "https://opengameart.org/content/american-asian-european-city-tilesets",
    "https://opengameart.org/content/rpg-desert-tileset",
    # ^ page 9
    "https://opengameart.org/content/16x16-castle-tiles",
    "https://opengameart.org/content/tiny-rpg-forest",
    "https://opengameart.org/content/some-8-bit-vertical-shooter-tiles-r2",
    "https://opengameart.org/content/2d-platform-winter-tiles",
    "https://opengameart.org/content/perspective-walls-template",
    "https://opengameart.org/content/track-tiles",
    "https://opengameart.org/content/isaiah658s-pixel-pack-2",
    "https://opengameart.org/content/space-merc-redux-giant-jungle-environment",
    "https://opengameart.org/content/collection-of-rune-stones-seamless-tiles",
    "https://opengameart.org/content/rpg-tilesets-pack",
    "https://opengameart.org/content/cave-tileset-4",
    "https://opengameart.org/content/old-frogatto-tile-art-2",
    "https://opengameart.org/content/race-track-tiles-0",
    "https://opengameart.org/content/stone-tower-defense-game-art",
    "https://opengameart.org/content/snowy-tiles",
    "https://opengameart.org/content/tropical-medieval-city-game-tileset",
    "https://opengameart.org/content/seamless-tileset-template-ii",
    "https://opengameart.org/content/432-isometrics-rocks-and-asteroids",
    "https://opengameart.org/content/wang-%E2%80%98blob%E2%80%99-tileset",
    "https://opengameart.org/content/platformer-tile-sheet",
    "https://opengameart.org/content/lpc-floors",
    "https://opengameart.org/content/terrain-elements",
    "https://opengameart.org/content/stone-home-exterior-tileset",
    "https://opengameart.org/content/32x32-grass-tile",
    "https://opengameart.org/content/isometric-painted-game-assets",
    "https://opengameart.org/content/wall-tiles",
    "https://opengameart.org/content/platform-tile-set-free",
    "https://opengameart.org/content/free-2d-block-forest-tile-pack",
    "https://opengameart.org/content/pixel-art-tiles-from-last-escape",
    "https://opengameart.org/content/pine-tree-tiles",
    "https://opengameart.org/content/plucky-girl-adventuror-animated",
    "https://opengameart.org/content/medieval-tileset",
    "https://opengameart.org/content/dungeon-tileset-32x32",
    "https://opengameart.org/content/dungeon-tile-set",
    "https://opengameart.org/content/land-tiles-v2",
    "https://opengameart.org/content/biomechanical-tile-sprite-sheet-001",
    "https://opengameart.org/content/autumn-platformer-tilesets",
    "https://opengameart.org/content/16x16-indoor-rpg-tileset",
    "https://opengameart.org/content/paper-pixels-8x8-platformer-assets",
    "https://opengameart.org/content/tiles-sands",
    "https://opengameart.org/content/stone-home-interior-tileset",
    "https://opengameart.org/content/platformer-grass-tileset",
    "https://opengameart.org/content/greenlands-tile-set-0",
    "https://opengameart.org/content/tileset-for-tile2map-with-tsx",
    "https://opengameart.org/content/monsters-villages-construction-set",
    "https://opengameart.org/content/16x16-wall-set",
    "https://opengameart.org/content/rpg-mansion-tile-set-nes",
    "https://opengameart.org/content/gems-4",
    "https://opengameart.org/content/basic-hex-tile-set-plus-16x16",
    "https://opengameart.org/content/greenlands-tile-set-orthographic",
    "https://opengameart.org/content/tileset-collection",
    "https://opengameart.org/content/sunnyland-forest-of-illusion",
    "https://opengameart.org/content/64x128-isometric-tiles-grassland-seasons",
    "https://opengameart.org/content/lpc-frama-sci-fi-extensions",
    "https://opengameart.org/content/pseudo-nes-tileset",
    "https://opengameart.org/content/pixel-art-street-and-avenue",
    "https://opengameart.org/content/c32-platformer-tiles-0",
    "https://opengameart.org/content/72x72-fantasy-tech-tileset",
    "https://opengameart.org/content/greenlands-tile-set-orthographic",
    "https://opengameart.org/content/painted-iso-roguelike-tiles",
    "https://opengameart.org/content/lpc-roofs",
    "https://opengameart.org/content/seamless-grass-texture",
    "https://opengameart.org/content/lpc-goes-to-space",
    "https://opengameart.org/content/nes-cc0-graphics-2",
    "https://opengameart.org/content/barricade-tiles",
    "https://opengameart.org/content/tile-set-style-gameboy",
    "https://opengameart.org/content/lpc-bazaar-rework",
    "https://opengameart.org/content/underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16",
    "https://opengameart.org/content/bountiful-bits-10x10-top-down-rpg-tiles",
    "https://opengameart.org/content/undead-pirate-roguelike",
    "https://opengameart.org/content/statues-fountains-collection",
    "https://opengameart.org/content/rpg-dungeon-package",
    "https://opengameart.org/content/4x4-dark-platform-tileset",
    "https://opengameart.org/content/lpc-caves",
    "https://opengameart.org/content/desert-tileset-1",
    "https://opengameart.org/content/seamless-tileset-template",
    "https://opengameart.org/content/beehive-interior-tileset",
    # ^ page 14
]


TILE_CONFIGS = [
    dict(name="1-bit-doomgeon-kit/doomgeonkit/DoomgeonKit/Characters-32x32.png", shape=(32, 32)),
    dict(name="1-bit-doomgeon-kit/doomgeonkit/DoomgeonKit/Tiles-16x16.png", shape=(16, 16)),
    dict(name="2d-lost-garden-zelda-style-tiles-resized-to-32x32-with-additions/mountain_landscape_23.png", shape=(32, 32)),
    dict(name="8x8-tileset-by-soundlust/soundsheett_vFinal.png", shape=(8, 8)),
    dict(name="16x16-game-assets/Base%20and%20Items/Items.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/2.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/3.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/4.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/7.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/8.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/9.png", shape=(32, 32)),
    dict(name="16x16-game-assets/tilesets_0/10.png", shape=(32, 32)),
    dict(name="16x16-snowy-town-tiles/snowtiles_1.gif", shape=(16, 16)),
    dict(name="32x32-fantasy-tileset/fantasy-tileset.png", shape=(32, 32)),
    dict(name="64x64-isometric-roguelike-tiles/rltiles-pack/64x64.png", shape=(64, 64)),
    dict(name="64x64-isometric-roguelike-tiles/rltiles-pack/catacomb.png", shape=(64, 64)),
    dict(name="64x64-isometric-roguelike-tiles/rltiles-pack/crypt.png", shape=(64, 64)),
    dict(name="64x64-isometric-roguelike-tiles/rltiles-pack/gallery.png", shape=(64, 64)),
    dict(name="64x64-isometric-roguelike-tiles/rltiles-pack/lair.png", shape=(64, 64)),
    dict(name="abandonauts-8x8-tile-assets/abandonauts_assets/tiles.png", shape=(8, 8)),
    dict(name="/home/bergi/prog/data/game-art/DOWN/a-blocky-dungeon/dungeon_sheet_0.png", shape=(16, 16)),
    dict(name="a-cute-dungeon/sheet_14.png", shape=(32, 32)),
    dict(name="basic-map-32x32-by-ivan-voirol/32x32_map_tile%20v3.1%20%5BMARGINLESS%5D.png", shape=(32, 32), ignore_lines=(0, 15), ignore_tiles=((16, 4), (16, 5), (16, 14), (16, 15))),
    dict(name="castle-tiles-for-rpgs/Castle2_5.png", shape=(32, 32)),
    dict(name="cave-tileset-0/cave.png", shape=(16, 16)),
    dict(name="/home/bergi/prog/data/game-art/DOWN/classical-ruin-tiles/classical_ruin_tiles_1.png", shape=(16, 16), offset=(0, 200)),
    dict(name="denzis-public-domain-art/DENZI_CC0_32x32_tileset.png", shape=(32, 32)),
    dict(name="desert-tileset-0/desert_1.png", shape=(16, 16)),
    dict(name="dungeon-crawl-32x32-tiles/ProjectUtumno_full_0.png", shape=(32, 32)),
    dict(name="dungeon-crawl-32x32-tiles/DungeonCrawl_ProjectUtumnoTileset.png", shape=(32, 32)),
    dict(name="dungeon-crawl-32x32-tiles-supplemental/ProjectUtumno_supplemental_0.png", shape=(32, 32)),
    dict(name="dungeon-crawl-32x32-tiles-supplemental/ProjectUtumno_full.png", shape=(32, 32)),
    dict(name="generic-platformer-tiles/generic_platformer_tiles_1.png", shape=(16, 16), offset=(0, 512 + 64 + 16)),
    dict(name="grass-and-water-tiles/grass_and_water.png", shape=(64, 64)),
    dict(name="instant-dungeon-v14-art-pack/instant_dungeon_artpack_v2_0/instant_dungeon_artpack/By Scott Matott/monsters.png", shape=(16, 16)),
    dict(name="instant-dungeon-v14-art-pack/instant_dungeon_artpack_v2_0/instant_dungeon_artpack/By Scott Matott/Players.png", shape=(16, 16)),
    dict(name="instant-dungeon-v14-art-pack/instant_dungeon_artpack_v2_0/instant_dungeon_artpack/By Scott Matott/weapons_spells_torch_key_gems.png", shape=(16, 16)),
    dict(name="isometric-64x64-medieval-building-tileset/iso-64x64-building_2.png", shape=(64, 64)),
    dict(name="isometric-64x64-outside-tileset/iso-64x64-outside.png", shape=(64, 64)),
    dict(name="lots-of-free-2d-tiles-and-sprites-by-hyptosis/hyptosis_tile-art-batch-1.png", shape=(32, 32)),
    dict(name="lots-of-free-2d-tiles-and-sprites-by-hyptosis/hyptosis_til-art-batch-2.png", shape=(32, 32)),
    dict(name="lots-of-free-2d-tiles-and-sprites-by-hyptosis/hyptosis_tile-art-batch-3_1.png", shape=(32, 32)),
    dict(name="lots-of-free-2d-tiles-and-sprites-by-hyptosis/hyptosis_tile-art-batch-4.png", shape=(32, 32)),
    dict(name="lpc-animated-water-and-fire/WaterAndFire_2.png", shape=(32, 32)),
    dict(name="lpc-farming-tilesets-magic-animations-and-ui-elements/submission_daneeklu/submission_daneeklu/tilesets/plowed_soil.png", shape=(32, 32)),
    dict(name="lpc-farming-tilesets-magic-animations-and-ui-elements/submission_daneeklu/submission_daneeklu/tilesets/sandwater.png", shape=(32, 32)),
    dict(name="lpc-farming-tilesets-magic-animations-and-ui-elements/submission_daneeklu/submission_daneeklu/tilesets/tallgrass.png", shape=(32, 32)),
    dict(name="lpc-terrain-repack/terrain/terrain.png", shape=(32, 32)),
    dict(name="lpc-terrains/lpc-terrains/lpc-terrains/terrain-map-v7.png", shape=(32, 32)),
    dict(name="lpc-terrains/lpc-terrains/lpc-terrains/terrain-v7.png", shape=(32, 32)),
    dict(name="lpc-tile-atlas/Atlas_0/base_out_atlas.png", shape=(16, 16)),
    dict(name="lpc-tile-atlas/Atlas_0/terrain_atlas.png", shape=(16, 16)),
    dict(name="lpc-tile-atlas2/Atlas2/build_atlas.png", shape=(16, 16)),
    dict(name="lpc-tile-atlas2/Atlas2/obj_misk_atlas.png", shape=(16, 16)),
    dict(name="metroid-like/metroid%20like.png", shape=(16, 16), limit=(128, 1000)),
    dict(name="minimalist-pixel-tileset/mininicular.png", shape=(8, 8)),
    dict(name="old-frogatto-tile-art/dirt-tiles.png", shape=(16, 16)),
    dict(name="orthographic-outdoor-tiles/tiles_12.png", shape=(16, 16)),
    dict(name="outdoor-tiles-again/sheet_19.png", shape=(16, 16)),
    dict(name="pixel-art-castle-tileset/castle_tileset_part1_0.png", shape=(16, 16)),
    dict(name="pixel-art-castle-tileset/castle_tileset_part2_0.png", shape=(16, 16)),
    dict(name="pixel-art-castle-tileset/castle_tileset_part3_0.png", shape=(32, 32)),
    dict(name="platformer-art-complete-pack-often-updated/Platformer%20Art%20Complete%20Pack_0/Base pack/Tiles/tiles_spritesheet.png", shape=(70, 70), offset=(1, 1), stride=(72, 72)),
    dict(name="platformer-art-complete-pack-often-updated/Platformer%20Art%20Complete%20Pack_0/Buildings expansion/sheet.png", shape=(70, 70)),
    dict(name="platformer-art-complete-pack-often-updated/Platformer%20Art%20Complete%20Pack_0/Candy expansion/sheet.png", shape=(70, 70)),
    dict(name="platformer-art-complete-pack-often-updated/Platformer%20Art%20Complete%20Pack_0/Ice expansion/sheet.png", shape=(70, 70)),
    dict(name="platformer-art-complete-pack-often-updated/Platformer%20Art%20Complete%20Pack_0/Request pack/sheet.png", shape=(70, 70)),
    dict(name="platformer-art-pixel-redux/Platformer%20Art%20Pixel%20Redux/spritesheet.png", shape=(21, 21), stride=(23, 23), offset=(2, 2)),
    dict(name="platform-pixel-art-assets/grotto_escape_pack/grotto_escape_pack/graphics/tiles.png", shape=(16, 16)),
    dict(name="platform-pixel-art-assets/grotto_escape_pack/grotto_escape_pack/graphics/player.png", shape=(16, 16)),
    dict(name="platform-pixel-art-assets/grotto_escape_pack/grotto_escape_pack/graphics/items.png", shape=(16, 16)),
    dict(name="platform-pixel-art-assets/grotto_escape_pack/grotto_escape_pack/graphics/enemies.png", shape=(16, 16)),
    dict(name="platform-tileset-nature/PlatformTiles_brownNature_ByEris_0/Tiles 64x64/Tiles_64x64.png", shape=(64, 64)),
    dict(name="post-apocalyptic-16x16-tileset-update1/apocalypse_0.png", shape=(16, 16)),
    dict(name="prototyping-2d-pixelart-tilesets/RPGTiles.png", shape=(32, 32)),
    dict(name="prototyping-2d-pixelart-tilesets/JnRTiles.png", shape=(32, 32)),
    dict(name="rpg-indoor-tileset-expansion-1/rpg%20indoor%20tileset%20expansion%201%20trans.png", shape=(16, 16)),
    dict(name="rpg-tiles-cobble-stone-paths-town-objects/PathAndObjects_0.png", shape=(32, 32)),
    dict(name="rpg-town-pixel-art-assets/town_rpg_pack/town_rpg_pack/graphics/transparent-bg-tiles.png", shape=(16, 16), ignore_tiles=((15, 15), (15, 16), (15, 17), (15, 18), (15, 19), (16, 15), (16, 16), (16, 17), (16, 18), (16, 19), (17, 15), (17, 16), (17, 17), (17, 18))),
    dict(name="rpg-town-pixel-art-assets/town_rpg_pack/town_rpg_pack/graphics/hero.png", shape=(16, 16)),
    dict(name="rpg-town-pixel-art-assets/town_rpg_pack/town_rpg_pack/graphics/npc.png", shape=(16, 16)),
    dict(name="sci-fi-platform-tiles/scifi_platformTiles_32x32.png", shape=(32, 32)),
    dict(name="seven-kingdoms/SevenKingdoms_graphics/SevenKingdoms_graphics/buildings/Village-sprites.png", shape=(32, 32)),
    dict(name="simple-broad-purpose-tileset/simples_pimples.png", shape=(16, 16), limit=(970, 10000)),
    dict(name="slates-32x32px-orthogonal-tileset-by-ivan-voirol/Slates%20v.2%20%5B32x32px%20orthogonal%20tileset%20by%20Ivan%20Voirol%5D_1.png", shape=(32, 32), offset=(32, 0)),
    dict(name="super-seasonal-platformer-tiles/Season_collection.png", shape=(16, 16), offset=(32, 0)),
    dict(name="the-field-of-the-floating-islands/tiles_packed_1.png", shape=(16, 16)),
    dict(name="the-field-of-the-floating-islands/snow-expansion.png", shape=(16, 16)),
    dict(name="tiny-16-basic/characters_1.png", shape=(16, 16)),
    dict(name="tiny-16-basic/things_0.png", shape=(16, 16)),
    dict(name="tiny-16-basic/basictiles_2.png", shape=(16, 16)),
    dict(name="tiny-16-basic/dead_1.png", shape=(16, 16)),
    dict(name="top-down-dungeon-tileset/sheet_17.png", shape=(16, 16)),
    dict(name="tuxemon-tileset/sheet_6.png", shape=(16, 16)),
    dict(name="victims-and-villagers/VictimsAndVillagers_0.png", shape=(16, 16)),
    dict(name="whispers-of-avalon-grassland-tileset/Extra_Unfinished4_1.png", shape=(32, 32), ignore_tiles=((7, 8), (7, 9), (7, 10),)),
    dict(name="whispers-of-avalon-grassland-tileset/Cliff_tileset_0.png", shape=(64, 64), limit=(500, 1000)),
    dict(name="whispers-of-avalon-grassland-tileset/object-%20layer_1.png", shape=(32, 32)),
    dict(name="worldmapoverworld-tileset/tileset_1.png", shape=(16, 16)),
    dict(name="zelda-like-tilesets-and-sprites/gfx_3/gfx/Overworld.png", shape=(16, 16)),
    dict(name="zelda-like-tilesets-and-sprites/gfx_3/gfx/objects.png", shape=(16, 16)),
    dict(name="zelda-like-tilesets-and-sprites/gfx_3/gfx/Inner.png", shape=(16, 16)),
    dict(name="zelda-like-tilesets-and-sprites/gfx_3/gfx/cave.png", shape=(16, 16)),
    dict(name="zoria-tileset/Zoria%20Tileset_1/overworld.png", shape=(16, 16)),
    dict(name="zoria-tileset/Zoria%20Tileset_1/underworld.png", shape=(16, 16)),
    dict(name="zoria-tileset/Zoria%20Tileset_1/sprites.png", shape=(16, 16)),
    dict(name="3x-updated-32x32-scifi-roguelike-enemies/ScifiCritters4_0.PNG", shape=(32, 32), stride=(33, 33)),
    dict(name="8-colors-4-stages-1-layer-64-16x16-tiles/Tile%20Map%208%20color_5.png", shape=(16, 16), offset=(33, 306), limit=(170, 1000)),
    dict(name="16-color-64-tiles-dungeon-01/Dungeons%20Tile%20Map%2016%20color.png", shape=(16, 16), offset=(68, 313), limit=(200, 1000)),
    dict(name="16x16-tiles/16x16-tiles_0.png", shape=(16, 16), stride=(17, 17)),
    dict(name="200-tileset-mega-metroidvania/BFT%20-%20Mega%20Metroidvania%20Tileset_8.png", shape=(16, 16)),
    dict(name="200-tileset-mega-metroidvania/A2.png", shape=(16, 16)),
    dict(name="alien-planet-platformer-tileset/alien-planet-tileset.png", shape=(32, 32)),
    dict(name="colored-16x16-fantasy-tileset/color_tileset_16x16_Jerom%26Eiyeron_CC-BY-SA-3.0_8.png", shape=(16, 16), limit=(432, 1000), ignore_tiles=((26, 7), (26, 8), (26, 9))),
    dict(name="dark-forest-town-tile-set-0/DarkHouseTiles_2.png", shape=(32, 32)),
    dict(name="denzis-32x32-orthogonal-tilesets/32x32_orthogonal_0/32x32 orthogonal/32x32_3quarters_view_Denzi060507.PNG", shape=(32, 32)),
    dict(name="denzis-32x32-orthogonal-tilesets/32x32_orthogonal_0/32x32 orthogonal/32x32_GUI_icons_Denzi050625-1.PNG", shape=(32, 32)),
    dict(name="denzis-32x32-orthogonal-tilesets/32x32_orthogonal_0/32x32 orthogonal/32x32_monochrome_walls_Denzi070708-2.PNG", shape=(32, 32)),
    dict(name="denzis-32x32-orthogonal-tilesets/32x32_orthogonal_0/32x32 orthogonal/32x32_monsters_Denzi120117-1.png", shape=(32, 32), offset=(32, 0)),
    dict(name="denzis-32x32-orthogonal-tilesets/32x32_orthogonal_0/32x32 orthogonal/32x32_spell_icons_Denzi090528-1.PNG", shape=(32, 32), offset=(32, 0)),
    dict(name="dutone-tileset-objects-and-character/minimalObjects_16x16Tiles.png", shape=(16, 16)),
    dict(name="goblin-caves/goblin_cave.png", shape=(32, 32), limit=(830, 1000)),
    dict(name="ground-tileset-grass-sand/ground_tilesest/ground.png", shape=(128, 128)),
    dict(name="lpc-submissions-merged/Animations/32x32-bat-sprite.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/beetle5.PNG", shape=(48, 48), offset=(2, 2), stride=(49, 49)),
    dict(name="lpc-submissions-merged/Animations/chicken_eat.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/chicken_walk.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/cow_eat.png", shape=(96, 96), stride=(128, 128), offset=(16, 16)),
    dict(name="lpc-submissions-merged/Animations/cow_walk.png", shape=(96, 96), stride=(128, 128), offset=(16, 16)),
    dict(name="lpc-submissions-merged/Animations/llama_eat.png", shape=(96, 96), stride=(128, 128), offset=(16, 16)),
    dict(name="lpc-submissions-merged/Animations/llama_walk.png", shape=(96, 96), stride=(128, 128), offset=(16, 16)),
    dict(name="lpc-submissions-merged/Animations/pig_eat.png", shape=(64, 64), stride=(128, 128), offset=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/pig_walk.png", shape=(64, 64), stride=(128, 128), offset=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/sheep_eat.png", shape=(64, 64), stride=(128, 128), offset=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/sheep_walk.png", shape=(64, 64), stride=(128, 128), offset=(32, 32)),
    dict(name="lpc-submissions-merged/Animations/WaterAndFire.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/LPC%20Submissions%20Merged%202.0/Exterior Tiles.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/LPC%20Submissions%20Merged%202.0/Interior.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/LPC%20Submissions%20Merged%202.0/Interior 2.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/LPC%20Submissions%20Merged%202.0/Outside Objects.png", shape=(32, 32)),
    dict(name="lpc-submissions-merged/LPC%20Submissions%20Merged%202.0/Terrain and Outside.png", shape=(32, 32)),
    dict(name="lpc-wooden-furniture/dark-wood_4.png", shape=(32, 32)),
    dict(name="lpc-wooden-furniture/blonde-wood_3.png", shape=(32, 32)),
    dict(name="lpc-wooden-furniture/green-wood.png", shape=(32, 32)),
    dict(name="lpc-wooden-furniture/white-wood.png", shape=(32, 32)),
    dict(name="micro-tileset-overworld-and-dungeon/Micro%20Tileset%20-%20Overworld%20%26%20Dungeon/tilesetP8.png", shape=(16, 16)),
    dict(name="multi-platformer-tileset-grassland-old-version/Multi_Platformer_Tileset_Free/Multi_Platformer_Tileset_Free/GrassLand/Terrain/Grassland_Terrain_Tileset.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/decorations-background.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/decorations-foreground.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/passable-rocks.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/rocks.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/rocks-foreground.png", shape=(16, 16)),
    dict(name="old-frogatto-tiles-pt2/old-frogatto-tiles2/walls.png", shape=(16, 16)),
    dict(name="pixelantasy/PIxelantasy%20-%20FREE_0/PIxelantasy - FREE/Tiles/tiles_001.png", shape=(16, 16)),
    dict(name="pixel-platformer-tile-set/Sprute.png", shape=(32, 32)),
    dict(name="platformer-pack-redux-360-assets/Platformer%20Pack%20Redux%20%28360%20assets%29/Spritesheets/spritesheet_complete.png", shape=(130, 130)),
    dict(name="retro-tileset/tilesets_7/A1_Master.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/A2_Master.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/A4_Master.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/A5_Master.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Dungeon_B.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Happy_New_Year.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Happy_New_Year_2.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Happy_New_Year_3.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Inside_B.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Outside_B.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Outside_C.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Outside_D.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Outside_E.png", shape=(48, 48)),
    dict(name="retro-tileset/tilesets_7/Outside_F.png", shape=(48, 48)),
    dict(name="space-merc-redux-platform-shmup-tileset/tiles-onemassivesheet-16colour-alpha_0.png", shape=(32, 32)),
    dict(name="space-shooter-sprites/spritesheet.png", shape=(32, 32)),
    dict(name="space-war-man-platform-shmup-set/oga-swm-tiles-alpha_0.png", shape=(8, 8), ignore_tiles=((14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (15, 28), (15, 29), (15, 30), (15, 31), (15, 32), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28), (16, 29), (16, 30), (16, 31), (16, 32), (17, 20), (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (17, 26), (17, 27), (17, 28), (17, 29), (17, 30), (17, 31), (17, 32), (15, 38), (15, 39), (15, 40), (15, 41), (15, 42), (15, 43), (16, 38), (16, 39), (16, 40), (16, 41), (16, 42), (15, 43),)),
    dict(name="steampunk-level-tileset-mega-pack-level-tileset-16x16/Steampunk%20Brick%20Back.png", shape=(16, 16)),
    dict(name="steampunk-level-tileset-mega-pack-level-tileset-16x16/Steampunk%20Blocks.png", shape=(16, 16)),
    dict(name="steampunk-level-tileset-mega-pack-level-tileset-16x16/Steampunk%20Corse%20Elements.png", shape=(16, 16)),
    dict(name="tileset-1bit-color-extention/extra-1bits_1.png", shape=(16, 16)),
    dict(name="1-layer-8-bit-15-color-4-stages-tileset/Tile%20Map%20Silver_1.png", shape=(16, 16), offset=(33, 306)),
    dict(name="4-color-dungeon-bricks-16x16/bricks.db32.png", shape=(16, 16)),
    dict(name="8-color-full-game-sprite-tiles/Tile%20Map%208%20color%20Full%20game.png", shape=(16, 16), offset=(16, 269), limit=(288, 470)),
    dict(name="animated-ocean-water-tile/Ocean_SpriteSheet.png", shape=(32, 32)),
    dict(name="castle-walls-isometric-64-x-128/TileObjectsRubbleWalls.png", shape=(64, 64)),
    dict(name="city-pixel-tileset/city_tileset/city_tileset/city.png", shape=(20, 20)),
    dict(name="cobblestone-tileset/cobbleset-64_0.png", shape=(64, 64)),
    dict(name="cobblestone-tileset/cobbleset-128.png", shape=(128, 128)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_monochrome_mixed_Denzi140330-12.png", shape=(16, 16), limit=(1000, 256), ignore_lines=(2, 14, 21, 27), ignore_tiles=((0, 5), (0, 6), (0, 7), (0, 8), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), )),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_monochrome_mixed_Denzi140330-12.png", shape=(16, 16), limit=(1000, 512), offset=(0, 256), ignore_lines=(0, 6, 11)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_retrostyle_monster_legacy_Denzi100701-1.png", shape=(16, 16), offset=(16, 32)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_ultim6_item_Denzi060817-2.PNG", shape=(16, 16)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_characters_simple_Denzi091025-5.PNG", shape=(16, 16), offset=(96, 48), limit=(1000, 256)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_character_animated_Denzi070812-2.PNG", shape=(16, 16), offset=(16, 0)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_modern_mixed_Denzi101114-6.png", shape=(16, 16), ignore_lines=(0, 1, 2, 8, 16, 19)),
    dict(name="denzis-16x16-orthogonal-tilesets/16x16_orthogonal_0/16x16 orthogonal/16x16_old_map_Denzi090523-1.PNG", shape=(16, 16), offset=(32, 32), ignore_tiles=((0, 8), (0, 9), (0, 19), (0, 20), )),
    dict(name="denzis-sidescroller-tilesets/denzi_sidescroller_0/sidescroller/scifi_effects_animated_Denzi130311-1.png", shape=(32, 32), offset=(64, 80)),
    dict(name="early-80s-arcade-pixel-art-dungeonsslimes-walls-power-ups-etc/spritesforyou.png", shape=(16, 16)),
    dict(name="exterior-32x32-town-tileset/tileset_town_multi_v002.png", shape=(32, 32)),
    dict(name="gameboy-rpg-tile/GameBoy%20Tile_0.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/monsters.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/torch_key_gems.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/stone_walls.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/WallTemplate-3and4-color.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/wizard_lightning_poof.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/Player.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/torch_key_gems.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/dracolich.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Scott Matott/RockyStoneWalls-no-center.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Voytek Falendysz/shield_knife_and_scrolls.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Voytek Falendysz/zombie.png", shape=(16, 16)),
    dict(name="instant-dungeon-art-pack/instant_dungeon_artpack_orig_0/instant_dungeon_artpack/By Voytek Falendysz/dragon.png", shape=(16, 16)),
    dict(name="lpc-alchemy/alchemy/alchemy/dark-wood.png", shape=(32, 32)),
    dict(name="lpc-alchemy/alchemy/alchemy/alchemy.png", shape=(32, 32), offset=(1024+32, 0)),
    dict(name="lpc-alchemy/alchemy/alchemy/alchemy.png", shape=(16, 16), limit=(1024, 1000)),
    dict(name="lpc-wooden-ship-tiles/ship_3.png", shape=(16, 16)),
    dict(name="map-tile/map_12.png", shape=(48, 48), stride=(49, 49), offset=(5, 3)),
    dict(name="mini-roguelike-8x8-tiles/miniroguelike-8x8_0.png", shape=(8, 8)),
    dict(name="modified-isometric-64x64-outside-tileset/terrain_0.png", shape=(64, 64)),
    dict(name="muckety-mudskipper-sprites-and-tiles/muckety_mudskipper_1/muckety_mudskipper/By Scott Matott/muckety_muddskipper_sprites_and_tiles_SMS.png", shape=(16, 16), limit=(230, 1000)),
    dict(name="puhzil/puhzil_0.png", shape=(16, 16), limit=(1000, 920)),
    dict(name="rock-tileset/RockTile.png", shape=(32, 32)),
    dict(name="roguelike-bosses/roguelikebosses.png", shape=(32, 32)),
    dict(name="rpg-tiles-%E2%80%94-forest-meadows-outdoor/8x8%20pixel%20tiles_4.png", shape=(8, 8)),
    dict(name="sokoban-100-tiles/kenney_sokobanPack/Preview_KenneyNL.png", shape=(36, 36), stride=(42, 42), offset=(9, 23)),
    dict(name="tiny16-tileset/tiny-16.png", shape=(16, 16)),
    dict(name="wyrmsun-cc0-over-900-items/wyrmsun_cc0/deep_water.png", shape=(32, 32)),
    dict(name="wyrmsun-cc0-over-900-items/wyrmsun_cc0/grass_and_dirt.png", shape=(32, 32)),
    dict(name="wyrmsun-cc0-over-900-items/wyrmsun_cc0/explosion.png", shape=(64, 64)),
    dict(name="wyrmsun-cc0-over-900-items/wyrmsun_cc0/big_fire.png", shape=(48, 48)),
    dict(name="wyrmsun-cc0-over-900-items/wyrmsun_cc0/ballista_warship.png", shape=(72, 72)),
    dict(name="4x4-dark-platform-tileset/fit%20a%20tileset%20in%2064x64%20pixels.png", shape=(8, 8)),
    dict(name="16x16-castle-tiles/16x16%20castle_0.png", shape=(16, 16)),
    dict(name="16x16-indoor-rpg-tileset/all_in_one.png", shape=(16, 16)),
    dict(name="16x16-wall-set/WallSet.png", shape=(16, 16)),
    dict(name="72x72-fantasy-tech-tileset/72x72/72x72_tiles.png", shape=(72, 72)),
    dict(name="barricade-tiles/barricade_tiles.png", shape=(128, 128)),
    dict(name="basic-hex-tile-set-plus-16x16/drjamgo_hexplus.png", shape=(16, 16)),
    dict(name="biomechanical-tile-sprite-sheet-001/biomechamorphs_001A.png", shape=(16, 16), stride=(17, 17), offset=(0, 1)),
    dict(name="bountiful-bits-10x10-top-down-rpg-tiles/bountiful-bits-v-2.1/Bountiful-Bits-10x10/Full.png", shape=(10, 10)),
    dict(name="c32-platformer-tiles-0/C32%20Platformer%20Tiles_0/Ground/Overworld.png", shape=(16, 16)),
    dict(name="c32-platformer-tiles-0/C32%20Platformer%20Tiles_0/Ground/Underworld.png", shape=(16, 16)),
    dict(name="c32-platformer-tiles-0/C32%20Platformer%20Tiles_0/Ground/UnderworldEx.png", shape=(16, 16)),
    dict(name="c32-platformer-tiles-0/C32%20Platformer%20Tiles_0/Ground/Nukalands.png", shape=(16, 16)),
    dict(name="c32-platformer-tiles-0/C32%20Platformer%20Tiles_0/Ground/NukalandsEx.png", shape=(16, 16)),
    dict(name="cave-tileset-4/cave_assets_1.png", shape=(16, 16), offset=(32, 0), limit=(1000, 550), ignore_tiles=((22, 1), (22, 2), (22, 3), (22, 4),)),
    dict(name="cave-tileset-4/cave_assets_1.png", shape=(16, 16), offset=(32, 544), limit=(200, 1000) ),
    dict(name="collection-of-rune-stones-seamless-tiles/rune-stones/rune stones/Stones-Tiled/Stone-Tile-4x4-256x256.jpg", shape=(64, 64)),
    dict(name="desert-tileset-1/desert_tileset32x32.png", shape=(32, 32)),
    dict(name="greenlands-tile-set-0/GreenlandsTileset.png", shape=(16, 16)),
    dict(name="greenlands-tile-set-orthographic/spr_Greenlands_orth_0.png", shape=(16, 16)),
    dict(name="greenlands-tile-set-orthographic/spr_Greenlands_orth_1.png", shape=(16, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Tileset/Tileset.png", shape=(16, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 1.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 11.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 3.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 6.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 7.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 8.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 9.png", shape=(20, 16)),
    dict(name="isaiah658s-pixel-pack-2/version_2.0_isaiah658s_pixel_pack_2/Characters/Character 11.png", shape=(20, 16)),
    dict(name="land-tiles-v2/DaLandv2.png", shape=(32, 32)),
    dict(name="lpc-caves/caves.png", shape=(32, 32)),
    dict(name="lpc-frama-sci-fi-extensions/scifi_hackspace_frama_0.png", shape=(16, 16), limit=(64, 1000)),
    dict(name="lpc-frama-sci-fi-extensions/scifi_hackspace_frama_0.png", shape=(16, 16), offset=(256, 0)),
    dict(name="lpc-goes-to-space/scifi_space_rpg_tiles_lpcized_0.png", shape=(16, 16)),
    dict(name="medieval-tileset/medieval%20tileset%20exterior.png", shape=(16, 16)),
    dict(name="medieval-tileset/medieval%20tileset%20interior.png", shape=(16, 16)),
    dict(name="monsters-villages-construction-set/monsters_villages_construction_set/Monsters Villages Construction Set/By Scott Matott/monster_village_tiles.png", shape=(16, 16)),
    dict(name="monsters-villages-construction-set/monsters_villages_construction_set/Monsters Villages Construction Set/By Scott Matott/monsters-all.png", shape=(16, 16)),
    dict(name="nes-cc0-graphics-2/NES_2.png", shape=(16, 16), stride=(18, 18), offset=(3, 3)),
    dict(name="perspective-walls-template/perspective_walls.png", shape=(64, 64)),
    dict(name="pine-tree-tiles/summer_pine_tree_tiles.png", shape=(32, 32)),
    dict(name="platformer-grass-tileset/grass_9.png", shape=(64, 64)),
    dict(name="platformer-tile-sheet/Platformer_tile-2018.11.14-1.png", shape=(16, 16), stride=(18, 18), offset=(351, 87), limit=(442, 122)),
    dict(name="platformer-tile-sheet/Platformer_tile-2018.11.14-1.png", shape=(16, 16), stride=(18, 18), offset=(351, 125), limit=(450, 195)),
    dict(name="platformer-tile-sheet/Platformer_tile-2018.11.14-1.png", shape=(16, 16), stride=(18, 18), offset=(351, 199), limit=(450, 280)),
    dict(name="platformer-tile-sheet/Platformer_tile-2018.11.14-1.png", shape=(16, 16), stride=(18, 18), offset=(351, 273), limit=(450, 380)),
    dict(name="pseudo-nes-tileset/nestileset.png", shape=(16, 16)),
    dict(name="rpg-dungeon-package/RPG%20DUNGEON%20VOL%201.png", shape=(16, 16), stride=(17, 17), ignore_tiles=((2, 7), (2, 8), (3, 7), (3, 8), (1, 5), (1, 6))),
    dict(name="rpg-dungeon-package/RPG%20DUNGEON%20VOL%202.png", shape=(16, 16), stride=(17, 17), ignore_tiles=((2, 7), (2, 8), (3, 7), (3, 8), (1, 5), (1, 6))),
    dict(name="rpg-dungeon-package/RPG%20DUNGEON%20VOL%203.png", shape=(16, 16), stride=(17, 17), ignore_tiles=((2, 7), (2, 8), (3, 7), (3, 8), (1, 5), (1, 6))),
    dict(name="rpg-mansion-tile-set-nes/NES%20Mansion%20Tile%20Set.png", shape=(16, 16), offset=(8, 8), ignore_tiles=((3, 15), (3, 16), (3, 17), (3, 18),)),
    dict(name="rpg-tilesets-pack/NessTilesPack/tiles/wooden_bridge_horizontal_rail.png", shape=(16, 16)),
    dict(name="rpg-tilesets-pack/NessTilesPack/tiles/temple_floor.png", shape=(16, 16)),
    dict(name="rpg-tilesets-pack/NessTilesPack/tiles/dark_tiles.png", shape=(16, 16)),
    dict(name="rpg-tilesets-pack/NessTilesPack/tiles/chess_tiles.png", shape=(16, 16)),
    dict(name="snowy-tiles/sprites_clean.gif", shape=(16, 16)),
    dict(name="space-merc-redux-giant-jungle-environment/fgtiles-alpha_0.png", shape=(32, 32)),
    dict(name="stone-home-interior-tileset/stone_house_interior.png", shape=(64, 64)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (1).png", shape=(32, 32)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (2).png", shape=(16, 16)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (3).png", shape=(32, 32)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (4).png", shape=(32, 32)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (5).png", shape=(16, 16)),
    dict(name="tileset-collection/sprite_sheets_0/sprite_sheet (6).png", shape=(16, 16)),
    dict(name="tileset-for-tile2map-with-tsx/tile2map_tileset_0/tile2map_tileset/tile2map64.png", shape=(64, 64)),
    dict(name="tile-set-style-gameboy/tile_setGameBoyStyle_5.png", shape=(32, 32)),
    dict(name="undead-pirate-roguelike/zombie-sheet.png", shape=(16, 16)),
    dict(name="undead-pirate-roguelike/goldbeard-sheet.png", shape=(32, 32)),
    dict(name="undead-pirate-roguelike/skeleton-sheet_0.png", shape=(16, 16)),
    dict(name="undead-pirate-roguelike/skeleton_with_hat-sheet.png", shape=(16, 16)),
    dict(name="undead-pirate-roguelike/zombie_eyepatch-sheet.png", shape=(16, 16)),
    dict(name="undead-pirate-roguelike/explosions_2.png", shape=(16, 16)),
    dict(name="undead-pirate-roguelike/icon_sheet.png", shape=(16, 16)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-atlas-32x32.png", shape=(32, 32)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-extra-32x32.png", shape=(32, 32)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-lomem-32x32.png", shape=(32, 32)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-sprites-16x16.png", shape=(16, 16)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-sprites-flameskull-32x32.png", shape=(32, 32)),
    dict(name="underworld-load-comprehensive-top-view-rpg-tileset-32x32-some-16x24-16x16/underworld_load_5/underworld_load/underworld_load-sprites-32x48.png", shape=(32, 32), offset=(16, 0), stride=(48, 32)),
    dict(name="wall-tiles/walls_x.png", shape=(32, 32)),
    dict(name="wang-%E2%80%98blob%E2%80%99-tileset/wang-blob-tilesets_0.png", shape=(32, 32), offset=(8, 278)),
]


def download_all(
        urls: List[str],
        root_path: str = STORAGE_DIRECTORY,
):
    from pathlib import Path

    import requests
    import bs4
    import os
    import zipfile

    for url in urls:
        name = url.split("/")[-1]

        folder = Path(root_path).expanduser() / name
        os.makedirs(folder, exist_ok=True)

        index_file = folder / "index.html"
        if not index_file.exists():
            print(f"downloading {url}")
            response = requests.get(url)
            index_file.write_text(response.text)

        markup = index_file.read_text()
        soup = bs4.BeautifulSoup(markup, features="html.parser")

        print(f"\n-- {url} --")
        div = soup.find("div", {"class": "field-name-field-art-files"})
        for a in div.find_all("a"):
            url = a.attrs["href"]
            filename = folder / url.split("/")[-1]

            if filename.suffix[1:].lower() not in ("png", "gif", "zip"):
                continue

            if not filename.exists():
                print(f"downloading {url}")
                response = requests.get(url)
                filename.write_bytes(response.content)

            if filename.suffix.lower() == ".zip":
                with zipfile.ZipFile(filename) as zipf:
                    for file in zipf.filelist:
                        if not file.is_dir():
                            fp = zipf.open(file.filename)
                            sub_filename = Path(str(filename)[:-4]) / file.filename
                            if not sub_filename.exists():
                                print(f"extracting {file.filename}")
                                os.makedirs(sub_filename.parent, exist_ok=True)
                                sub_filename.write_bytes(fp.read())


class RpgTileBootstrapIterableDataset(IterableDataset):

    def __init__(
            self,
            tile_configs: List[dict],
            shape: Tuple[int, int, int] = (3, 32, 32),
            directory: str = STORAGE_DIRECTORY,
            interleave: bool = False,
    ):
        self.shape = shape
        self.directory = directory
        self.interleave = interleave
        self.tilesets = tile_configs

    def __iter__(self):
        if not self.interleave:
            for params in self.tilesets:
                yield from self._iter_tiles(**params)
        else:
            iterables = [
                self._iter_tiles(**params)
                for params in self.tilesets
            ]
            while iterables:
                next_iterables = []
                for it in iterables:
                    try:
                        yield next(it)
                        next_iterables.append(it)
                    except StopIteration:
                        pass
                iterables = next_iterables

    def _iter_tiles(
            self,
            name: str,
            shape: Tuple[int, int],
            offset: Tuple[int, int] = None,
            stride: Optional[Tuple[int, int]] = None,
            limit: Optional[Tuple[int, int]] = None,
            remove_transparent: bool = True,
            ignore_lines: Iterable[int] = None,
            ignore_tiles: Iterable[Tuple[int, int]] = None,
    ):
        if ignore_lines:
            ignore_lines = set(ignore_lines)
        if ignore_tiles:
            ignore_tiles = set(ignore_tiles)

        image = PIL.Image.open(
            (Path(self.directory) / name).expanduser()
        )
        if image.mode == "P":
            image = image.convert("RGBA")
        image = VF.to_tensor(image)

        if image.shape[0] != self.shape[0]:
            if image.shape[0] == 4 and remove_transparent:
                image = image[:3] * image[3].unsqueeze(0)

            image = set_image_channels(image[:3], self.shape[0])

        if limit:
            image = image[..., :limit[0], :limit[1]]
        if offset:
            image = image[..., offset[0]:, offset[1]:]

        for patch, pos in iter_image_patches(image, shape, stride=stride, with_pos=True):
            pos = tuple(int(p) // s for p, s in zip(pos, shape))

            if ignore_lines and pos[0] in ignore_lines:
                continue
            if ignore_tiles and pos in ignore_tiles:
                continue

            if patch.std(1).mean() > 0.:
                patch = VF.resize(patch, self.shape[-2:], VF.InterpolationMode.NEAREST, antialias=False)
                config = {
                    "name": name,
                    "shape": shape,
                    "offset": offset,
                    "stride": stride,
                    "limit": limit,
                    "ignore_lines": list(ignore_lines) if ignore_lines else None,
                    "ignore_tiles": list(ignore_tiles) if ignore_tiles else None,
                }
                yield patch, config, pos


def filter_dataset(
        tile_configs: List[dict],
        output_filename: Union[str, Path] = Path(__file__).resolve().parent / "tile-config.ndjson"
):
    ds = RpgTileBootstrapIterableDataset(
        tile_configs=tile_configs,
        shape=(1, 16, 16),
    )
    ds_dissimilar = DissimilarImageIterableDataset(
        ds, max_similarity=.99, yield_bool=True,
        verbose=True,
    )

    count = 0
    used_files = set()
    ignore_tiles = {}
    try:
        for patch, dissimilar, config, pos in ds_dissimilar:

            # a single file might be used multiple times
            #   so we need to consider the slicing config as well
            key = (config["name"], json.dumps(config))

            if not dissimilar:
                count += 1
                used_files.add(key)
            else:
                if key not in ignore_tiles:
                    ignore_tiles[key] = []
                ignore_tiles[key].append(pos)

    except KeyboardInterrupt:
        pass

    with open(output_filename, "wt") as fp:
        for key in sorted(used_files):
            filename, config = key
            config = json.loads(config)

            if key in ignore_tiles:
                if not config["ignore_tiles"]:
                    config["ignore_tiles"] = []
                # merge the ignored tiles from dissimilarity test
                config["ignore_tiles"] = set(config["ignore_tiles"]) | set(ignore_tiles[key])

            if config["ignore_tiles"]:
                config["ignore_tiles"] = sorted(config["ignore_tiles"])

            config = {
                key: value for key, value in config.items()
                if value is not None
            }
            print(f"\n{config}")
            print(json.dumps(config), file=fp)

    print(f"dataset tiles: {count:,}")
    print(f"ignored tiles: {sum(len(i) for i in ignore_tiles.values()):,}")
    print(f"dataset files: {len(used_files):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str,
        choices=["download", "filter"],
    )
    args = parser.parse_args()

    if args.command == "download":
        download_all(OPEN_GAME_ART_URLS)

    elif args.command == "filter":
        filter_dataset(
            tile_configs=TILE_CONFIGS,
        )

