from pathlib import Path
from io import StringIO
from typing import Any

import marko
import github_slugger

LOG_PATH = Path(__file__).resolve().parent


def update_readme():

    slugger = github_slugger.GithubSlugger()

    fp = StringIO()
    for file in sorted(LOG_PATH.rglob("*.md")):
        if file.name != "README.md":
            render_file_index(file, slugger, fp)

    fp.seek(0)

    readme = (LOG_PATH / "README.md").read_text()
    readme = readme[:readme.index("## Index\n") + 9] + fp.read()
    (LOG_PATH / "README.md").write_text(readme)
    print(readme)


class IndexRenderer(marko.Renderer):

    def __init__(self, file_path: Path, slugger: github_slugger.GithubSlugger):
        super().__init__()
        self.file_path = file_path
        self.slugger = slugger

    def render_children(self, element: Any) -> Any:
        if isinstance(element, str):
            return element
        return super().render_children(element)

    def render_heading(self, element: marko.block.Heading) -> str:
        title = self.render_children(element)
        return "{indent}- [:arrow_forward:]({link}) {title}".format(
            indent="  " * element.level,
            title=title,
            link=f"{self.file_path}#{self.slugger.slug(title)}"
        )


def render_file_index(file: Path, slugger: github_slugger.GithubSlugger, out: StringIO):

    print(f"\n\n- [{file.name}]({file.relative_to(LOG_PATH)})", file=out)

    doc = marko.Markdown().parse(file.read_text())
    for child in doc.children:
        if child.get_type() == "Heading":
            renderer = IndexRenderer(file.relative_to(LOG_PATH), slugger)
            print(renderer.render(child), file=out)


if __name__ == "__main__":
    update_readme()
