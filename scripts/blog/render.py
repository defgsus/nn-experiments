from pathlib import Path
from typing import Dict, Any

import marko
import marko.inline
from jinja2 import Environment as JinjaEnvironment
from jinja2 import FileSystemLoader
from marko import inline, block
from marko.ext.gfm.renderer import GFMRendererMixin


def render_document_html(
        document: marko.block.Document,
        link_mapping: Dict[str, str],
):
    return HTMLRenderer(link_mapping=link_mapping).render(document)


def render_teaser_html(
        document: marko.block.Document,
):
    return HTMLTeaserRenderer().render(document)


def render_template(
        markup: str,
        context: dict,
):
    from .__main__ import TEMPLATE_PATH

    env = JinjaEnvironment(
        loader=FileSystemLoader(searchpath=[TEMPLATE_PATH]),
    )

    template = env.from_string(
        source=markup,
    )
    return template.render(**context)


class HTMLRenderer(GFMRendererMixin, marko.HTMLRenderer):

    def __init__(self, link_mapping: Dict[str, str]):
        super().__init__()
        self._link_mapping = link_mapping

    def render_heading(self, element: marko.block.Heading) -> str:
        if hasattr(element, "anchor"):
            return (
                """<h{level} id="{slug}">{children} <a href="#{slug}" class="heading-linker">←</a></h{level}>\n"""
            ).format(
                level=element.level,
                children=self.render_children(element),
                slug=element.anchor.slug,
            )
        return super().render_heading(element)

    def render_link(self, element: marko.inline.Link) -> str:
        dest = self._link_mapping.get(element.dest, element.dest)
        external = "//" in dest
        template = '<a href="{}"{}{}>{}</a>'
        title = f' title="{self.escape_html(element.title)}"' if element.title else ""
        target = f' target="_blank"' if external else ""
        url = self.escape_url(dest)
        body = self.render_children(element)
        return template.format(url, title, target, body)

    def render_image(self, element: marko.inline.Image) -> str:
        element.dest = self._link_mapping.get(element.dest, element.dest)
        html = super().render_image(element)
        return f"""<div style="overflow: scroll;">{html}</div>"""

    def render_table(self, element):
        html = super().render_table(element)
        return f"""<div style="overflow: scroll;">{html}</div>"""


class HTMLTeaserRenderer(GFMRendererMixin, marko.HTMLRenderer):

    def __init__(self):
        super().__init__()
        self._paragraphs = 0

    def render_children(self, element: Any) -> Any:
        if self._paragraphs >= 2:
            return ""
        return super().render_children(element)

    def render_heading(self, element: marko.block.Heading) -> str:
        if self._paragraphs > 0:
            return self.render_children(element)
        return ""

    def render_image(self, element: marko.inline.Image) -> str:
        return ""

    def render_table(self, element):
        return ""

    def render_link(self, element: inline.Link) -> str:
        return self.render_children(element)

    def render_paragraph(self, element):
        if self._paragraphs >= 2:
            return ""
        self._paragraphs += 1
        return super().render_paragraph(element)

    def render_code_block(self, element: block.CodeBlock) -> str:
        return ""

    def render_list(self, element: block.List) -> str:
        return ""

    def render_fenced_code(self, element: block.FencedCode) -> str:
        return ""

    def render_quote(self, element: inline.Literal) -> str:
        return self.render_children(element)

    def render_thematic_break(self, element: block.ThematicBreak) -> str:
        return ""
