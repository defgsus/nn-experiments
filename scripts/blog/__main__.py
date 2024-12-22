import argparse
import os
import sys
from pathlib import Path
from io import StringIO
from typing import List, Tuple

from scripts.blog.document import Document
from scripts.blog.sitemap import Sitemap
from docs.logs.update_index import update_readme

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
DOCS_PATH = PROJECT_PATH / "docs"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates"


def main(
        command: str,
):
    documents: List[Document] = []

    for file in sorted(DOCS_PATH.rglob("*.md")):
        if file.name != "README.md":
            documents.append(Document.from_file(file))

    sitemap = Sitemap(documents)
    sitemap.create_index_page()

    if command == "dump":
        sitemap.dump()

    elif command == "test":
        update_readme(do_write=False)
        sitemap.render_all(write=False)
        print()
        print("Call with 'render' to actually render files")

    elif command == "render":
        update_readme()
        sitemap.render_all(write=True)

    else:
        raise NotImplementedError(f"unknown command {command}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, default="test", nargs="?",
        choices=["test", "render", "dump"],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
