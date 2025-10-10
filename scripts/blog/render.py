import re
from pathlib import Path
import html as html_lib
from typing import Dict, Any, List, Tuple

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
    return HTMLRenderer(
        link_mapping=link_mapping,
    ).render(document)


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


def _get_renderer_base_class():
    md = marko.Markdown(extensions=["gfm", "codehilite"])
    md._setup_extensions()
    return type(md.renderer)


class HTMLRenderer(_get_renderer_base_class()):

    def __init__(self, link_mapping: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self._link_mapping = link_mapping
        self._generated_id = 1022

    def generate_id(self):
        self._generated_id += 1
        return f"subgenius-{self._generated_id}"

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

    def render_fenced_code(self, element: marko.block.FencedCode):
        if element.lang.startswith("llmchat"):
            return self._render_llm_chat(element)
        if element.lang:
            html = super().render_fenced_code(element)
        else:
            html = f"<pre>{html_lib.escape(element.children[0].children)}</pre>"
        return f"""<div style="overflow: scroll;">{html}</div>"""

    def _render_llm_chat(self, element: marko.block.FencedCode):
        all_open = "+all" in element.lang
        system_open = "+system" in element.lang
        tools_open = "+tools" in element.lang
        chat_text: str = element.children[0].children
        sections = _split_chat_text(chat_text)
        htmls = []
        for role, text in sections:
            checkbox_id = self.generate_id()
            block_open = (
                all_open
                or role in ("user", "assistant", "tool_call")
                or (role == "system" and system_open)
                or (role == "available_tools" and tools_open)
            )
            htmls.append(
                """
                <label for="{checkbox_id}">
                    <div class="llm-chat-window llm-chat-role-{role}">
                        <input id="{checkbox_id}" type="checkbox" {checked_attr}>
                        <div class="llm-chat-window-title">{role}</div>
                        <div class="llm-chat-window-content">{html}</div>
                    </div>
                </label>
                """.format(
                    role=role,
                    html=self._render_llm_chat_block(text),
                    checkbox_id=checkbox_id,
                    checked_attr='checked=""' if block_open else "",
                )
            )

        html = """
            <div class="llm-chat">
                {windows}
                <label class="llm-chat-full-transcript">
                    <input id="{checkbox_id}" type="checkbox">
                    <div class="llm-chat-label-show">Show plain text</div>
                    <pre>{full_transcript}</pre>
                </label>
            </div>
        """.format(
            windows="\n".join(htmls),
            checkbox_id=self.generate_id(),
            full_transcript=html_lib.escape(chat_text.replace("\\```", "```")),
        )
        # print("X", html)
        return html

    def _render_llm_chat_block(self, text: str) -> str:
        blocks = re.split(r"\\```", text)
        is_code_block = False
        htmls = []
        for block in blocks:
            block = html_lib.escape(block)
            if is_code_block:
                html = f"""<pre>```{block.rstrip()}\n```</pre>"""
            else:
                html = block.strip()
            htmls.append(html)
            is_code_block = not is_code_block
        return "".join(htmls)

def _split_chat_text(text: str) -> List[Tuple[str, str]]:
    special_tokens = {
        "<|start_of_role|>": "role",
        "<|end_of_role|>": "text",
        "<|end_of_text|>": "text",
        "<|tool_call|>": "tool_call",
    }
    parsed_text = ""
    blocks = [{"block": "text", "text": ""}]
    for c in text:
        parsed_text += c

        is_new_block = False
        for special_token, next_block_type in special_tokens.items():
            if parsed_text.endswith(special_token):
                # hack: catch special case in system prompt
                if special_token == "<|tool_call|>" and parsed_text.endswith("respond only with <|tool_call|>"):
                    continue
                blocks[-1]["text"] = blocks[-1]["text"][:-len(special_token)+1]
                if blocks[-1]["block"] == "role":
                    blocks[-1]["text"] += f"|{len(blocks)}"
                blocks.append({"block": next_block_type, "text": ""})
                is_new_block = True
                break

        if not is_new_block:
            blocks[-1]["text"] += c

    sections = []
    section_type = "raw"
    for block in blocks:
        if not block["text"].strip():
            continue

        if block["block"] == "role":
            section_type = block["text"]
        elif block["block"] != "text":
            section_type = block["block"]

        if not sections or sections[-1]["type"] != section_type:
            sections.append({"type": section_type.split("|")[0], "text": ""})

        if block["block"] != "role":
            sections[-1]["text"] += block["text"]

    return [(s["type"], s["text"]) for s in sections if s["text"]]





class HTMLTeaserRenderer(GFMRendererMixin, marko.HTMLRenderer):

    def __init__(self):
        super().__init__()
        self._paragraphs = 0

    def render_children(self, element: Any) -> Any:
        if self._paragraphs >= 2:
            return ""
        return super().render_children(element)

    def render_heading(self, element: marko.block.Heading) -> str:
        #if self._paragraphs > 0:
        #    return self.render_children(element)
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

    def render_html_block(self, element):
        return ""