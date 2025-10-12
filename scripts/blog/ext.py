import re
from typing import Tuple, Dict

from marko import block, inline
from marko.helpers import MarkoExtension


def extract_markdown_foot_notes(text: str) -> Tuple[str, Dict[str, str]]:
    foot_notes = {}
    foot_note_index = 1
    cur_start = None
    last_end = 0
    trimmed_text = []
    for match in re.finditer(r"(\[\[\[start|end]]])", text, re.MULTILINE):
        if match.group() == "[[[start":
            if cur_start is not None:
                raise ValueError(f"`[[[start` tag at {match.span()} duplicates tag at {cur_start}")
            cur_start = match.span()[0]
        elif match.group() == "end]]]":
            if cur_start is None:
                raise ValueError(f"`end]]]` tag at {match.span()} without at `[[[start`")
            end = match.span()[1]
            foot_notes[str(foot_note_index)] = text[cur_start + 8: end - 6]
            trimmed_text.append(text[last_end:cur_start])
            trimmed_text.append(f"[[[footnote-{foot_note_index}]]]")
            last_end = end
            cur_start = None
            foot_note_index += 1

    trimmed_text.append(text[last_end:])
    return "".join(trimmed_text), foot_notes


class IntegratedFootnoteReference(inline.InlineElement):

    pattern = r"\[\[\[footnote-(\d+)]]]"
    parse_children = False

    def __init__(self, match):
        super().__init__(match)
        self.children = []
        self.footnote_id = match.groups()[0]


class IntegratedFootnoteRenderMixin(object):

    def render_integrated_footnote_reference(self, element: IntegratedFootnoteReference):
        return f"[{element.footnote_id}]"


IntegratedFootnoteExtension = MarkoExtension(
    elements=[IntegratedFootnoteReference],
    renderer_mixins=[IntegratedFootnoteRenderMixin]
)
