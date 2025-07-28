import os
from pathlib import Path
import argparse
import base64
import io
from pprint import pprint
from typing import Dict, List

import PIL.Image
import nbformat
import marko


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename", type=str,
        help="filename of notebook",
    )

    parser.add_argument(
        "post_filename", type=str,
        help="filename of post, e.g. docs/posts/2025/2025-01-01-name",
    )

    return vars(parser.parse_args())


def main(
        filename: str,
        post_filename: str,
):
    post_filename = Path(post_filename)
    asset_sub_path = Path(f"assets/{post_filename.with_suffix('').name}")

    image_map: Dict[Path, dict] = {}

    sfile = io.StringIO()

    nb = nbformat.reads(Path(filename).read_text(), 4)
    for cell in nb["cells"]:

        if cell["cell_type"] == "markdown":
            print(cell["source"], end="\n\n", file=sfile)

        image_basename = "image"
        for output in cell.get("outputs", []):
            if output.get("data"):
                has_image_in_this_cell = False
                for key in output["data"].keys():
                    if key == "text/html":
                        print(f'\n{output["data"][key]}\n', file=sfile)

                    elif key == "text/plain":
                        text = output["data"][key]
                        if text.startswith("'image:"):
                            image_basename = text[7:-1]

                    elif key.startswith("image/"):
                        if has_image_in_this_cell:
                            continue
                        else:
                            has_image_in_this_cell = True
                        image_format = key[6:]
                        image_encoded = output["data"].get(key)
                        image_bytes = base64.b64decode(image_encoded.encode())
                        if image_format != "png":
                            image_bytes = io.BytesIO(image_bytes)
                            image_pil = PIL.Image.open(image_bytes, formats=[image_format])
                            image_bytes = io.BytesIO()
                            image_pil.save(image_bytes, format="png")
                            image_bytes.seek(0)
                            image_bytes = image_bytes.read()

                        image_key = image_basename
                        c = 2
                        while image_key in image_map:
                            image_key = f"{image_basename}-{c:02}"
                            c += 1

                        image_map[image_key] = {
                            "filename": (post_filename.parent / asset_sub_path / f"{image_key}.png").relative_to(
                                post_filename.parent
                            ),
                            "data": image_bytes,
                        }

                        alt = image_basename.replace("-", " ")
                        print(f"![{alt}]({image_map[image_key]['filename']})\n", file=sfile)

            #pprint(output["data"].keys())

    for img in image_map.values():
        filename = post_filename.parent / img["filename"]
        print("writing image:", filename)
        os.makedirs(filename.parent, exist_ok=True)
        filename.write_bytes(img["data"])

    sfile.seek(0)
    text = sfile.read()
    print("writing post:", post_filename)
    post_filename.write_text(text)


if __name__ == "__main__":
    main(**parse_args())

