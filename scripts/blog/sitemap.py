import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from io import StringIO
from typing import List, Tuple, Dict, Optional, Union, Generator, Any

import marko
import scss

from .document import Document


class Page:
    def __init__(
            self,
            url: Path,
            document: Document,
            front_matter: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.document = document
        self.front_matter = front_matter or {}

    def __repr__(self):
        return f"Page('{self.url}')"

    @property
    def title(self) -> str:
        if self.front_matter.get("title"):
            return self.front_matter["title"]
        if self.document.anchors and self.document.anchors[0].level == 1:
            return self.document.anchors[0].title
        return self.document.filename.with_suffix("").name

    def relative_url(self, filename: Union[str, Path]) -> str:
        """
        Make the filename/url relative to this page.

        :param filename: a path that starts at docs/ directory, e.g. logs/img/someimage.png
        :return: a relative url, e.g. ../logs/img/someimage.png
        """
        return (
            "/".join(".." for p in Path(self.url).parents if str(p) != ".")
            + f"/{filename}"
        ).lstrip("/")

    @property
    def scss_files(self) -> List[str]:
        return ["style.scss"]

    @property
    def template(self) -> str:
        return self.front_matter.get("template", "article")

    @property
    def teaser(self) -> str:
        from .render import render_teaser_html
        return render_teaser_html(self.document.document)

    @property
    def date(self) -> datetime.date:
        if re.match(r"^\d\d\d\d-\d\d-\d\d", self.document.filename.name):
            return datetime.datetime.strptime(self.document.filename.name[:10], "%Y-%m-%d").date()
        return datetime.date(2000, 1, 1)


class StyleSheet:

    def __init__(self, original_filename: Path):
        self.original_filename = original_filename
        self._content = None

    @property
    def target_filename(self) -> str:
        name = self.original_filename.with_suffix(".css").name
        #hash = hashlib.md5(self.content().encode()).hexdigest()[:8]
        return f"html/style/{name}"

    def content(self):
        from .__main__ import TEMPLATE_PATH

        if self._content is None:
            compiler = scss.Compiler(
                root=TEMPLATE_PATH,
                ignore_parse_errors=False,
            )
            self._content = compiler.compile(self.original_filename)
        return self._content


class Sitemap:

    def __init__(
            self,
            documents: List[Document],
    ):
        from .__main__ import TEMPLATE_PATH, DOCS_PATH

        self.template_path = TEMPLATE_PATH
        self.docs_path = DOCS_PATH

        self.base_templates = {
            "base": (self.template_path / "base.html").read_text(),
            "index": (self.template_path / "index.html").read_text(),
            "article": (self.template_path / "article.html").read_text(),
        }
        self.stylesheets_mapping = {
            "style.scss": StyleSheet(self.template_path / "style.scss"),
        }

        self._doc_page_mapping: Optional[Dict[str, Page]] = None
        self._pages: List[Page] = []

        for doc in sorted(documents, key=lambda doc: doc.filename.name):

            relative_filename = doc.filename.relative_to(self.docs_path)
            url = self._markdown_filename_to_page_url(relative_filename)

            if list(filter(lambda p: p.url == url, self._pages)):
                raise AssertionError(f"Multiple page url '{url}'")

            self._pages.append(
                Page(
                    url=url,
                    document=doc,
                )
            )

        self._pages.sort(key=lambda p: p.date)

    def _markdown_filename_to_page_url(self, filename: Path):
        name = filename.with_suffix(".html").name
        if re.match(r"^\d\d\d\d-\d\d-\d\d-", name):
            name = name[11:]
        return Path(f"html") / filename.parent / name

    def render_page(self, page: Page) -> str:
        from .render import render_document_html, render_template

        markup = render_document_html(
            page.document.document,
            link_mapping=self.page_link_mapping(page),
        )
        pages = list(self.iter_pages())
        page_index = pages.index(page)
        context = {
            "html": {
                "title": page.title,
                "body": markup,
                "css": [
                    page.relative_url(self.get_stylesheet(f).target_filename)
                    for f in page.scss_files
                ],
            },
            "page": page,
            "pages": list(reversed(pages)),
            "previous_page": pages[page_index - 1] if page_index > 0 else None,
            "next_page": pages[page_index + 1] if page_index < len(pages) - 1 else None,
        }
        html = render_template(self.base_templates[page.template], context)

        return html

    def create_index_page(self):
        from .__main__ import DOCS_PATH
        # make up a document, just to be able to use `Page` below
        doc = Document(
            filename=DOCS_PATH / "index.md",
            document=marko.block.Document(),
            anchors=[],
            links=[],
        )
        page = Page(url=Path("index.html"), document=doc, front_matter={
            "title": "Home",
            "template": "index",
        })
        self._pages.insert(0, page)

    def iter_pages(self) -> Generator[Page, None, None]:
        yield from self._pages

    def dump(self):
        for page in self.iter_pages():
            print(page)
            for link, mapped_link in self.page_link_mapping(page).items():
                print(f"  {link} -> {mapped_link}")

    def page_link_mapping(self, page: Page) -> Dict[str, str]:
        link_mapping: Dict[str, str] = {}
        for base_link in page.document.links:
            if "//" in base_link:
                continue

            link = page.document.link_mapping[base_link]
            slug = None
            if "#" in link:
                link, slug = link.split("#", 1)

            if link.endswith(".md"):
                linked_page = self.page_by_doc_filename(link)
                mapped_link = str(linked_page.url)

            else:
                mapped_link = link

            if "//" not in mapped_link:
                mapped_link = page.relative_url(mapped_link)

            if slug is not None:
                mapped_link = f"{mapped_link}#{slug}"

            link_mapping[base_link] = mapped_link

        return link_mapping

    def page_by_doc_filename(self, doc_name: Union[str, Path]) -> Page:
        if self._doc_page_mapping is None:
            doc_map = {}
            for page in self.iter_pages():
                doc_map[str(page.document.relative_filename)] = page
            self._doc_page_mapping = doc_map

        return self._doc_page_mapping[str(doc_name)]

    def get_stylesheet(self, scss_file: str) -> StyleSheet:
        return self.stylesheets_mapping[scss_file]

    def render_all(self, write: bool):
        from .__main__ import DOCS_PATH

        def _write_file(filename: Union[str, Path], content):
            filename = DOCS_PATH / str(filename).lstrip("/")
            exists = filename.exists()
            unchanged = exists and filename.read_text() == content

            if not unchanged and write:
                os.makedirs(filename.parent, exist_ok=True)
                filename.write_text(content)

            if not exists:
                tag = "CREATED  "
            else:
                tag = 'unchanged' if unchanged else 'CHANGED  '
            print(f"{tag}: {filename}")

        for page in self.iter_pages():
            _write_file(page.url, self.render_page(page))

        for style in self.stylesheets_mapping.values():
            _write_file(style.target_filename, style.content())
