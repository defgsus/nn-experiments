import secrets
import uuid
from pathlib import Path
from typing import Union, Dict, Tuple

import nbconvert
from nbformat.notebooknode import NotebookNode


def notebook_to_markdown(
        filename: Union[str, Path],
) -> Tuple[str, Dict[str, bytes], Dict[str, str]]:
    preprocessor = PreProc()
    exporter = nbconvert.MarkdownExporter(preprocessors=[preprocessor])
    print(f"reading notebook {filename.name}")
    markdown, resources = exporter.from_filename(str(filename))

    # remove empty blocks
    markdown = markdown.replace("```python\n\n```", "")

    preprocessor.placeholder_map['<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG">'] = ""
    return markdown, preprocessor.image_map, preprocessor.placeholder_map


class PreProc(nbconvert.preprocessors.Preprocessor):
    """
    Handles cell visibility and collects images
    """

    def __init__(self, default_show_code: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.show_code = default_show_code
        self.image_map = {}
        self.placeholder_map = {}

    def preprocess_cell(self, cell: NotebookNode, resources: nbconvert.exporters.ResourcesDict, index):
        #print("CELL", cell, index)
        if cell.cell_type == "code":

            for i, output in enumerate(cell.get("outputs") or []):
                if data := output.get("data"):
                    if data.get("text/html"):
                        placeholder = f"<!-- PLACEHOLDER-{uuid.uuid4()} -->"
                        self.placeholder_map[placeholder] = data["text/html"]
                        data["text/html"] = placeholder

                    #if data.get("application/vnd.plotly.v1+json"):
                    #    del data["application/vnd.plotly.v1+json"]

            hide = cell.source.startswith("#hide\n")
            hide_cell = cell.source.startswith("#hide-cell\n")
            hide_output = cell.source.startswith("#hide-output\n")

            if hide or hide_cell or hide_output:
                cell.source = "\n".join(cell.source.splitlines()[1:])

            if hide or hide_output:
                cell.outputs = []
                cell.execution_count = None
            if hide or hide_cell:
                cell.source = ""

        for output in (cell.get("outputs") or []):
            if data := output.get("data"):
                if data.get("image/png"):
                    original_filename = output.metadata.filenames["image/png"]
                    filename = f"{resources['metadata']['name']}_files/{original_filename}"
                    output.metadata.filenames["image/png"] = filename
                    self.image_map[filename] = resources["outputs"][original_filename]

        return cell, resources
