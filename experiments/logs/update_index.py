import warnings
from pathlib import Path
from io import StringIO
from typing import Any, List, Tuple

import marko
import github_slugger

LOG_PATH = Path(__file__).resolve().parent


def update_readme():

    documents = []

    fp = StringIO()
    for file in sorted(LOG_PATH.rglob("*.md")):
        if file.name != "README.md":
            documents.append((file, render_file_index(file, fp)))

    # check for wrong links
    has_error = False
    for file, (doc, slugs) in documents:
        renderer = LinkCheckRenderer(file, documents)
        renderer.render(doc)

        if renderer.errors:
            print(f"\nInvalid links in {file}:")
            for error in renderer.errors:
                print(f" - {error}")

            has_error = True

    if has_error:
        print("\nrejecting your request!")
        exit(-1)

    fp.seek(0)

    readme = (LOG_PATH / "README.md").read_text()
    readme = readme[:readme.index("## Index\n") + 9] + fp.read()
    (LOG_PATH / "README.md").write_text(readme)
    print(readme)


class IndexRenderer(marko.Renderer):
    """
    Customized renderer for headings and links
    """
    def __init__(self, file_path: Path, slugger: github_slugger.GithubSlugger):
        super().__init__()
        self.file_path = file_path
        self.slugger = slugger
        self.slugs = []

    def render_children(self, element: Any) -> Any:
        if isinstance(element, str):
            return element
        return super().render_children(element)

    def render_heading(self, element: marko.block.Heading) -> str:
        title = self.render_children(element)
        slug = self.slugger.slug(title)
        self.slugs.append(slug)
        return "{indent}- [:arrow_forward:]({link}) {title}".format(
            indent="  " * element.level,
            title=title,
            link=f"{self.file_path}#{slug}"
        )


class LinkCheckRenderer(marko.HTMLRenderer):

    def __init__(
            self,
            file: Path,
            documents: List[Tuple[Path, Tuple[marko.block.Document, List[str]]]],
    ):
        super().__init__()
        self.file = file
        self.documents = documents
        self.errors = []

    def render_link(self, element: marko.block.inline.Link):
        url = element.dest

        if not url.startswith("http"):

            if url.startswith("#"):
                self._check_slug(self.file, url[1:])

            else:
                slug = None
                if "#" in url:
                    url, slug = url.split("#")

                file = self.file.parent / url
                if not file.exists():
                    self.errors.append(url)
                else:
                    if slug:
                        self._check_slug(file, slug)

        return super().render_link(element)

    def render_image(self, element: marko.block.inline.Image):
        url = element.dest

        if not url.startswith("http"):

            file = self.file.parent / url
            if not file.exists():
                self.errors.append(url)

        return super().render_image(element)

    def _check_slug(self, file: Path, slug: str):
        for doc in self.documents:
            if doc[0] == file:
                slugs = doc[1][1]
                if slug not in slugs:
                    self.errors.append(f"{file}#{slug}")
                return

        warnings.warn(f"A file with slug is referenced which is not in index: {file}#{slug}")


def render_file_index(
        file: Path,
        out: StringIO,
) -> Tuple[marko.block.Document, List[str]]:
    """
    Renders the index of markdown file to the `out` buffer,
    returns marko Document and list of found heading slugs
    """
    print(f"\n\n- [{file.name}]({file.relative_to(LOG_PATH)})", file=out)

    doc = marko.Markdown().parse(file.read_text())

    slugger = github_slugger.GithubSlugger()
    slugs = []
    for child in doc.children:
        # grab each heading and render it with the IndexRenderer
        if child.get_type() == "Heading":
            renderer = IndexRenderer(file.relative_to(LOG_PATH), slugger)
            print(renderer.render(child), file=out)

            # also collect the slugs in this document
            slugs.extend(renderer.slugs)

    return doc, slugs


if __name__ == "__main__":
    update_readme()
