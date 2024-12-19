import os
from pathlib import Path
from io import StringIO
from typing import List, Tuple

from scripts.blog.document import Document
from scripts.blog.sitemap import Sitemap
from docs.logs.update_index import update_readme

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
DOCS_PATH = PROJECT_PATH / "docs"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates"


def update_blog():
    documents: List[Document] = []

    for file in sorted(DOCS_PATH.rglob("*.md")):
        if file.name != "README.md":
            documents.append(Document.from_file(file))

    sitemap = Sitemap(documents)
    sitemap.create_index_page()
    #sitemap.dump()
    sitemap.render_all(write=True)


if __name__ == "__main__":
    update_readme()
    update_blog()
