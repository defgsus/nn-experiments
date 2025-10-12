import dataclasses
import io
from pathlib import Path
from typing import List, Any, Tuple, Dict, Optional

import marko
import marko.inline
import github_slugger
import yaml
from marko import block, inline

from .nbooks import notebook_to_markdown
from .ext import IntegratedFootnoteExtension, IntegratedFootnoteReference, extract_markdown_foot_notes


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
            frontmatter: Optional[dict] = None,
            asset_map: Optional[Dict[str, bytes]] = None,
            placeholder_map: Optional[Dict[str, str]] = None,  # replace in final html
            foot_notes: Optional[Dict[str, block.Element]] = None,
    ):
        from .__main__ import DOCS_PATH
        self.filename = filename
        self.relative_filename = filename.relative_to(DOCS_PATH)
        assert ".." not in str(self.relative_filename), self.relative_filename
        self.document = document
        self.anchors = anchors
        self.links = links
        self.frontmatter = frontmatter or dict()
        self.asset_map = asset_map
        self.placeholder_map = placeholder_map
        self.foot_notes = foot_notes
        self._link_mapping: Optional[Dict[str, str]] = None
        self.assets = [
            link for link in links
            if "//" not in link and link.rsplit(".", 1)[-1] in ("png", "csv")
        ]

    def __repr__(self):
        return f"Document('{self.relative_filename}')"

    @classmethod
    def from_file(cls, filename: Path):
        if filename.suffix.lower() == ".md":
            markdown, asset_map, placeholder_map = filename.read_text(), None, None
        elif filename.suffix.lower() == ".ipynb":
            markdown, asset_map, placeholder_map = notebook_to_markdown(filename)
        else:
            raise NotImplementedError(f"Filetype unhandled: {filename}")

        frontmatter, text = split_front_matter_and_markup(markdown)

        text, foot_notes = extract_foot_notes(text)

        parser = marko.Markdown(
            extensions=['gfm', 'codehilite', IntegratedFootnoteExtension]
        )
        doc = parser.parse(text)

        slugger = github_slugger.GithubSlugger()

        renderer = ExtractionRenderer(slugger, foot_notes)
        renderer.render(doc)

        return cls(
            filename=filename,
            document=doc,
            anchors=renderer.anchors,
            links=renderer.links,
            frontmatter=frontmatter,
            asset_map=asset_map,
            placeholder_map=placeholder_map,
            foot_notes=foot_notes,
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

                if link:
                    filename = (self.filename.parent / link)
                else:
                    filename = self.filename

                if not filename.exists():
                    if not self.asset_map or link not in self.asset_map:
                        raise AssertionError(
                            f"Document {self} links to non-existent file '{link}' (resolved: '{filename.resolve()}')"
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
                    link = f"{link}#{slug}"
                    mapped_link = f"{mapped_link}#{slug}"

                self._link_mapping[link] = str(mapped_link)

        return self._link_mapping


class ExtractionRenderer(marko.HTMLRenderer):
    """
    extracts all relevant content like anchors and links
    """
    def __init__(self, slugger: github_slugger.GithubSlugger, foot_notes: Dict[str, block.Element]):
        super().__init__()
        self.slugger = slugger
        self.foot_notes = foot_notes
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

    def render_integrated_footnote_reference(self, element: IntegratedFootnoteReference):
        """Render integrated footnotes in-place, just to catch any contained resources"""
        doc = self.foot_notes[element.footnote_id]
        return self.render(doc)


def split_front_matter_and_markup(markup: str) -> Tuple[Optional[dict], str]:
    """
    Split a text into it's front-matter and markup

    :return: tuple of (dict|None, str)
    """
    markup_lines = markup.strip().splitlines()
    if len(markup_lines) < 4:
        return None, "\n".join(markup_lines) + "\n"

    if markup_lines[0].strip() != "---":
        return None, "\n".join(markup_lines) + "\n"

    fm_lines = []
    markup_lines.pop(0)
    while markup_lines and markup_lines[0] != "---":
        fm_lines.append(markup_lines.pop(0))

    if not markup_lines:
        return None, "\n".join(markup_lines) + "\n"

    returned_markup = "\n".join(markup_lines[1:]) + "\n"
    front_matter = "\n".join(fm_lines) + "\n"

    fp = io.StringIO(front_matter)
    front_matter = yaml.safe_load(fp)

    return front_matter, returned_markup


def extract_foot_notes(text: str) -> Tuple[str, Dict[str, block.Element]]:
    text, foot_notes = extract_markdown_foot_notes(text)
    # parse the footnotes separately
    parser = marko.Markdown(extensions=['gfm', 'codehilite'])
    for footnote_id, footnote_text in foot_notes.items():
        foot_notes[footnote_id] = doc = parser.parse(footnote_text)
        #for elem in iter_marko_sub_elements(doc):
        #    if isinstance(elem, block.BlockElement):
        #        if not isinstance(elem, (block.Document, block.Paragraph)):
        #            raise ValueError(f"Found block element in foot-note '{footnote_id}': {elem}")

    return text, foot_notes


def iter_marko_sub_elements(element: block.Element):
    yield element
    try:
        for e in element.children:
            yield from iter_marko_sub_elements(e)
    except AttributeError:
        pass
