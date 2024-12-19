import dataclasses
from pathlib import Path
from typing import List, Any, Tuple, Dict, Optional

import marko
import marko.inline
import github_slugger
from marko import block, inline
from marko.ext.gfm import gfm


class Document:

    @dataclasses.dataclass
    class Anchor:
        title: str
        slug: str
        level: int

    def __init__(
            self,
            filename: Path,
            document: marko.block.Document,
            anchors: List[Anchor],
            links: List[str],
    ):
        from .__main__ import DOCS_PATH
        self.filename = filename
        self.relative_filename = filename.relative_to(DOCS_PATH)
        assert ".." not in str(self.relative_filename), self.relative_filename
        self.document = document
        self.anchors = anchors
        self.links = links
        self._link_mapping: Optional[Dict[str, str]] = None
        self.assets = [
            link for link in links
            if "//" not in link and link.rsplit(".", 1)[-1] in ("png", "csv")
        ]

    def __repr__(self):
        return f"Document('{self.relative_filename}')"

    @classmethod
    def from_file(cls, filename: Path):

        doc = gfm.parse(filename.read_text())

        slugger = github_slugger.GithubSlugger()

        renderer = ExtractionRenderer(slugger)
        renderer.render(doc)

        return cls(
            filename=filename,
            document=doc,
            anchors=renderer.anchors,
            links=renderer.links,
        )

    @property
    def link_mapping(self) -> Dict[str, str]:
        from .__main__ import DOCS_PATH

        if self._link_mapping is None:
            self._link_mapping = {}
            for link in self.links:

                if "//" in link:
                    continue

                slug = None
                if "#" in link:
                    link, slug = link.split("#", 1)

                filename = (self.filename.parent / link)
                if not filename.exists():
                    raise AssertionError(
                        f"Document {self} links to non-existent file '{link}' (resolved: '{filename}')"
                    )

                if "../src/" in str(filename):
                    short = str(filename).split("../src/", 1)[-1]
                    mapped_link = f"https://github.com/defgsus/nn-experiments/blob/master/src/{short}"

                elif "../scripts/" in str(filename):
                    short = str(filename).split("../scripts/", 1)[-1]
                    mapped_link = f"https://github.com/defgsus/nn-experiments/blob/master/scripts/{short}"

                else:
                    # resolve ../ links
                    mapped_link = (DOCS_PATH / str(filename.relative_to(DOCS_PATH))).resolve().relative_to(DOCS_PATH)
                    assert ".." not in str(mapped_link), f"mapped_link={mapped_link}, filename={filename}"

                if slug is not None:
                    mapped_link = f"{mapped_link}#{slug}"

                self._link_mapping[link] = str(mapped_link)

        return self._link_mapping

    def iter_elements(self):
        yield from self._iter_elements(self.document)

    def _iter_elements(self, element: marko.block.Element):
        for c in element.children:
            yield c
        yield element


class ExtractionRenderer(marko.HTMLRenderer):
    """
    extracts all relevant content like anchors and links
    """
    def __init__(self, slugger: github_slugger.GithubSlugger):
        super().__init__()
        self.slugger = slugger
        self.anchors = []
        self.links = []

    # grab heading anchors
    def render_heading(self, element: marko.block.Heading) -> str:
        title = self.render_children(element)
        slug = self.slugger.slug(title)
        anchor = Document.Anchor(
            title=title, slug=slug, level=element.level
        )
        element.anchor = anchor
        self.anchors.append(anchor)
        return super().render_heading(element)

    def render_link(self, element: inline.Link) -> str:
        self.links.append(element.dest)
        return super().render_link(element)

    def render_image(self, element: inline.Image) -> str:
        self.links.append(element.dest)
        return super().render_image(element)
