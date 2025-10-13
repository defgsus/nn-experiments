import re
from typing import Tuple, Dict

from marko import block, inline
from marko.helpers import MarkoExtension


def extract_markdown_foot_notes(text: str) -> Tuple[str, Dict[str, str]]:
    foot_notes = {}
    foot_note_index = 1
    start_positions = []
    last_end = 0
    trimmed_text = []
    for match in re.finditer(r"(\[\[\[start|end]]])", text, re.MULTILINE):
        if match.group() == "[[[start":
            start_positions.append(match.span()[0])
        elif match.group() == "end]]]":
            if not start_positions:
                raise ValueError(f"`end]]]` tag at {match.span()} without at `[[[start`")
            end = match.span()[1]
            foot_notes[str(foot_note_index)] = text[start_positions[-1] + 8: end - 6]
            trimmed_text.append(text[last_end:start_positions[-1]])
            trimmed_text.append(f"[[[footnote-{foot_note_index}]]]")
            last_end = end
            foot_note_index += 1
            start_positions.pop(-1)

    if start_positions:
        raise ValueError(f"Unclosed [[[start at {start_positions[-1]}")

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
