import json
import os
import subprocess
from pathlib import Path


BUNKER_COMMAND = Path("~/prog/nk/bunker/bunker").expanduser()
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "cache" / "babel-transpile"


def transpile_js(source: str) -> str:
    if not CACHE_PATH.exists():
        os.makedirs(CACHE_PATH)
        subprocess.check_call(
            [str(BUNKER_COMMAND), "npm", "install", "--save-dev", "@babel/preset-env", "babel@/cli"],
            cwd=CACHE_PATH,
        )

        (CACHE_PATH / "babel.config.json").write_text(json.dumps({
            "presets": [["@babel/preset-env", {"useBuiltIns": False}]]
        }, indent=2))

    (CACHE_PATH / "source.js").write_text(source)

    subprocess.check_call(
        [str(BUNKER_COMMAND), "./node_modules/@babel/cli/bin/babel", "source.js", "-o", "bundle.js"],
        cwd=CACHE_PATH,
    )

    return (CACHE_PATH / "bundle.js").read_text()
