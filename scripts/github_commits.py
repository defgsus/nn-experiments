"""


creating clip embedding, date=2023-11-20T03:19:10Z:  11%|████                                 | 188800/1724733 [34:41<4:38:38, 91.87it/s]
"""
import argparse
import datetime
import gzip
import itertools
import json
import base64
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.util import to_torch_device
from src.util.gharchive import GHArchive
from src.util.files import iter_ndjson, iter_lines
from src.util.embedding import normalize_embedding
from src.console import CC


def parse_args():
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "--path", type=str, nargs="?", default=str(Path("~/prog/data/gharchive").expanduser())
    )

    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser("show", help="Get commit messages from gharchive.org and dump")
    parser.set_defaults(command="show")

    def _add_date_params(parser, with_days: bool = True):
        parser.add_argument(
            "date", type=str,
            help="start date of push-events, YYYY-MM-DD"
        )
        if with_days:
            parser.add_argument(
                "days", type=int, nargs="?", default=1,
                help="Number of days to process since 'date'"
            )

    def _add_text_filter_params(parser):
        parser.add_argument(
            "--min-text-length", type=int, nargs="?", default=0,
            help="consider texts with at least that number of characters",
        )

    _add_date_params(parser)
    _add_text_filter_params(parser)

    parser = subparsers.add_parser("get", help="Get commit messages from gharchive.org")
    parser.set_defaults(command="get")

    _add_date_params(parser)
    _add_text_filter_params(parser)

    parser = subparsers.add_parser("embed", help="Get the embeddings for the commit messages")
    parser.set_defaults(command="embed")

    def _add_model_params(parser):
        parser.add_argument(
            "--model", type=str, nargs="?", default="clip/ViT-B/32",
            help="The model used to encode text, can be `clip/<name>` or a huggingface transformer"
                 ", like 'thenlper/gte-small', 'sentence-transformers/all-MiniLM-L6-v2'"
                 ", check https://huggingface.co/spaces/mteb/leaderboard for a list of models sorted by size"
        )
        parser.add_argument(
            "--trust-remote-code", type=bool, nargs="?", default=False, const=True,
            help="Trust remote code when downloading huggingface models",
        )

    _add_date_params(parser)
    _add_text_filter_params(parser)
    _add_model_params(parser)

    parser = subparsers.add_parser("query", help="Query for similar commit messages in interactive console")
    parser.set_defaults(command="query")

    _add_date_params(parser, with_days=False)
    _add_text_filter_params(parser)
    _add_model_params(parser)

    parser = subparsers.add_parser("plot", help="Render html output")
    parser.set_defaults(command="plot")

    _add_date_params(parser, with_days=False)
    _add_text_filter_params(parser)
    _add_model_params(parser)
    parser.add_argument(
        "--limit", type=int, nargs="?", default=100_000,
        help="The number of message to plot in the tSNE map"
    )

    return vars(main_parser.parse_args())


def main(
        command: str,
        **kwargs
):
    command_func = globals().get(f"command_{command}")
    if not callable(command_func):
        print(f"Invalid command '{command}'")
        exit(1)

    command_func(**kwargs)


def get_filenames(path, date, min_text_length, model=""):
    path = Path(path)
    mt = f"-tl{min_text_length}" if min_text_length >= 0 else ""

    if model.startswith("clip/"):
        mn = model
    else:
        mn = f"hug-{model}"

    mn = mn.replace('/', '-')

    class Filenames:
        commits = path / f"commits-{date}{mt}.ndjson.gz"
        info = path / f"commits-{date}{mt}.json"
        embeddings = path / f"commits-{date}{mt}-{mn}.b64.gz"
        html = Path(f"commits-{date}{mt}-{mn}.html")
    return Filenames


def command_show(
        path: str,
        date: str,
        days: int,
        min_text_length: int,
):
    gharchive = GHArchive(verbose=True, raw_path=path)
    commit_iterator = GHArchiveCommitIterator(gharchive)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for day in range(days):

        date = start_date + datetime.timedelta(days=day)
        filenames = get_filenames(path, date, min_text_length)

        print(f"creating file {filenames.commits} ...")
        os.makedirs(filenames.commits.parent, exist_ok=True)

        with gzip.open(filenames.commits, "wt") as fp:
            for commit in commit_iterator.iter_commits(start_date=date, days=1, min_text_length=min_text_length):
                print(repr(commit["message"]))


def command_get(
        path: str,
        date: str,
        days: int,
        min_text_length: int,
):
    gharchive = GHArchive(verbose=True, raw_path=path)
    commit_iterator = GHArchiveCommitIterator(gharchive)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for day in range(days):

        date = start_date + datetime.timedelta(days=day)
        filenames = get_filenames(path, date, min_text_length)

        print(f"creating file {filenames.commits} ...")
        os.makedirs(filenames.commits.parent, exist_ok=True)

        total = 0
        abort = False
        try:
            with gzip.open(filenames.commits, "wt") as fp:
                for commit in commit_iterator.iter_commits(start_date=date, days=1, min_text_length=min_text_length):
                    fp.write(json.dumps(commit, separators=(',', ':')) + "\n")
                    total += 1

        except KeyboardInterrupt:
            print("aborted")
            abort = True

        filenames.info.write_text(json.dumps({
            "date": str(date),
            "total": total,
        }, indent=2))

        if abort:
            break


def command_embed(
        path: str,
        date: str,
        days: int,
        model: str,
        trust_remote_code: bool,
        min_text_length: int,
        batch_size: int = 128,
):
    import torch
    from src.util.text_encoder import TextEncoder

    encoder = TextEncoder(model_name=model, trust_remote_code=trust_remote_code)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    try:
        for day in range(days):

            date = start_date + datetime.timedelta(days=day)
            filenames = get_filenames(path, date, min_text_length, model)

            if not filenames.commits.exists():
                print("NOT FOUND:", filenames.commits)
                continue

            info = json.loads(filenames.info.read_text())

            print(f"creating file {filenames.embeddings} ...")
            os.makedirs(filenames.embeddings.parent, exist_ok=True)

            def _iter_text_batches():
                text_batch = []
                with tqdm(total=info["total"]) as progress:
                    for commit in iter_ndjson(filenames.commits):
                        text_batch.append(commit["message"])
                        if len(text_batch) >= batch_size:
                            yield text_batch
                            text_batch.clear()

                        progress.update(1)
                        progress.desc = f"creating clip embedding, date={commit['date']}"

                if text_batch:
                    yield text_batch

            with gzip.open(filenames.embeddings, "wt") as fp:
                for texts in _iter_text_batches():
                    with torch.no_grad():
                        embeddings = encoder.encode(texts=texts).cpu().half().numpy()

                        for embedding in embeddings:
                            fp.write(
                                base64.b64encode(embedding.data).decode("ascii") + "\n"
                            )

    except KeyboardInterrupt:
        print("aborted")


def command_query(
        path: str,
        date: str,
        model: str,
        min_text_length: int,
        trust_remote_code: bool,
):
    import numpy as np
    import torch
    from src.util.text_encoder import TextEncoder

    encoder = TextEncoder(model_name=model)

    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    commits, embeddings = load_data(path, date, min_text_length, model)

    print(f"loaded embeddings: {embeddings.shape}")
    print()

    try:
        while True:
            query = input("query> ").strip()
            if not query:
                continue

            if "not:" in query:
                queries = [q.strip() for q in query.split("not:")]
            else:
                queries = [query]

            with torch.no_grad():
                embedding = encoder.encode(queries).cpu().half().numpy()

            similarities: np.ndarray = (embedding @ embeddings.T)
            if similarities.shape[0] > 1:
                similarities[0] -= .5 * similarities[1]

            similarities = similarities[0]

            best_indices = similarities.argsort()

            print()
            for idx in best_indices[-20:]:
                sim = similarities[idx]
                c = commits[idx]
                print(f"{CC.rgb(1., 1., 1.)}  {sim:.3f} https://github.com/{c['repo']}{CC.Off}")
                print(f"        {repr(c['message'][:2000])}")
            print()

    except KeyboardInterrupt:
        print()


def command_plot(
        path: str,
        date: str,
        model: str,
        trust_remote_code: bool,
        min_text_length: int,
        limit: int,
):
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import plotly
    import plotly.express as px
    plotly.io.templates.default = "plotly_dark"

    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    filenames = get_filenames(path, date, min_text_length, model)
    commits, embeddings = load_data(path, date, min_text_length, model, limit)

    print(f"loaded embeddings: {embeddings.shape}")
    print(f"building tSNE...")

    reducer = TSNE(2, verbose=True, perplexity=20)
    positions = reducer.fit_transform(embeddings[:limit])

    print(f"clustering ...")
    clusterer = KMeans(50, n_init="auto")
    labels = clusterer.fit_predict(embeddings[:limit])

    df = pd.DataFrame({
        "x": positions[:limit, 0],
        "y": positions[:limit, 1],
        "repo": [c["repo"] for c in commits[:limit]],
        "sha": [c["sha"] for c in commits[:limit]],
        "message": [c["message"][:1000].replace("</script>", "") for c in commits[:limit]]
    })
    fig = px.scatter(
        df,
        x="x", y="y",
        height=600,
        hover_data=["repo"],
        custom_data=["repo", "sha", "message"],
        color=[str(c) for c in labels],
    )

    html = fig.to_html(
        full_html=False,
        include_mathjax=False,
        include_plotlyjs=True,
        post_script="""
            function commit_html(data) {
                const [repo, sha, message] = data;
                let html = `<h4><a href="https://github.com/${repo}">${repo}</a>/<a href="https://github.com/${repo}/commit/${sha}">${sha}</a></h4>`;
                html += `<pre>${message}</pre>`;
                return html;
            }

            
            // attach to plotly
            document.getElementById("{plot_id}").on("plotly_click", function(click_data) {
                console.log(click_data);
                const html = commit_html(click_data.points[0].customdata);
                document.querySelector("#below-plot").innerHTML = html;
            });
            document.getElementById("{plot_id}").on("plotly_selected", function(click_data) {
                console.log(click_data);
                const html = [];
                click_data.points.forEach(function(p) {
                    html.push(commit_html(p.customdata));
                });
                document.querySelector("#below-plot").innerHTML = html.join(" ");
            });
            
        """
    )
    filenames.html.write_text("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta
            http-equiv="Content-Security-Policy"
            content="default-src 'self' 'unsafe-inline' 'unsafe-eval'; script-src 'self' 'unsafe-inline' 'unsafe-eval';" 
        >
        <style type="text/css">
            body {
                background: #202020;
                color: #d0d0d0;
                font-family: mono;
            }
            a {
                color: #8989e3;
            }
        </style>
    </head>
    <body>
        %(html)s
        <div id="below-plot"></div>
    </body>
    </html>
    """ % {"html": html})




def iter_base64(filename, dtype=None):
    import numpy as np

    if dtype is None:
        dtype = np.float16

    for line in iter_lines(filename):
        embedding = np.frombuffer(base64.b64decode(line), dtype=dtype)
        yield embedding


def load_data(path: str, date: datetime.date, min_text_length: int, model: str, limit: Optional[int] = None):
    import numpy as np

    filenames = get_filenames(path, date, min_text_length, model)
    info = json.loads(filenames.info.read_text())

    commits = []
    embeddings = []
    try:
        for commit, embedding in tqdm(zip(
                iter_ndjson(filenames.commits),
                iter_base64(filenames.embeddings)
        ), total=info["total"], desc=f"loading {filenames.embeddings.name}"):
            commits.append(commit)
            embeddings.append(embedding[None, ...])

            if limit is not None and len(commits) >= limit:
                break

    except KeyboardInterrupt:
        print("loading aborted")

    embeddings = normalize_embedding(np.concatenate(embeddings))

    return commits, embeddings


class GHArchiveCommitIterator:
    """
    Iterates through the gharchive commits
    and keeps a list commit messages to filter out duplicates.
    """
    def __init__(self, gharchive: GHArchive, message_buffer_size: int = 1_000_000):
        self.gharchive = gharchive
        self.message_buffer_size = message_buffer_size
        self.message_dict = {}

    def iter_commits(
            self,
            start_date: datetime.date = datetime.date(2023, 11, 20),
            days: int = 7,
            min_text_length: int = 0,
    ):
        iterables = [
            self.gharchive.iter_events(
                day=start_date + datetime.timedelta(i),
                event_type="PushEvent",
            )
            for i in range(days)
        ]
        iterable = itertools.chain(*iterables)

        num_skipped = 0
        num_yielded = 0

        with tqdm() as progress:
            for event in iterable:
                #print(json.dumps(event, indent=2))
                data = {
                    "repo": event["repo"]["name"],
                    "date": event["created_at"],
                }
                for commit in event["payload"]["commits"]:
                    message = commit["message"]

                    if min_text_length > 0 and len(message) < min_text_length:
                        num_skipped += 1
                        continue

                    if not self.is_message_accepted(message) or message in self.message_dict:
                        num_skipped += 1
                        continue

                    self.message_dict[message] = num_yielded

                    yield {
                        **data,
                        "sha": commit["sha"],
                        "message": commit["message"],
                    }
                    num_yielded += 1

                progress.update(1)
                progress.desc = (
                    f"messages/skips {num_yielded:,}/{num_skipped:,}"
                    f", buffer-size {len(self.message_dict):,}"
                    f", date={data['date']}"
                )

                if len(self.message_dict) >= self.message_buffer_size:
                    median = sorted(self.message_dict.values())
                    # print("min/median/max", median[0], median[len(median) // 2], median[-1])
                    median = median[len(median) // 2]

                    self.message_dict = {
                        msg: step
                        for msg, step in self.message_dict.items()
                        if step <= median
                    }

    @classmethod
    def is_message_accepted(cls, text: str) -> bool:
        if " " not in text:
            return False

        text_l = text.lower()
        for skip_start in cls.SKIP_STARTS:
            if text_l.startswith(skip_start):
                return False

        for skip_text in cls.SKIP_TEXT:
            if skip_text in text_l:
                return False

        for c in text_l:
            code = ord(c)
            if 0x300 <= code < 0x2000 or code >= 0x4000:
                return False

        return True

    SKIP_STARTS = [
        "update from http",
        "revert \"",
        "merge branch ",
        "adding: ",
        "merge pull request #",
        "merging ",
        "merge commit ",
        "bump ",
        "{\"message\"",
        "bot:",
        "chore:",
        "build:",
        "flat:",
        "ci(deps):",
        "chore(deps):",
        "build(deps):",
        "github:",
        "[DEVOPS]",
        "modified:",
        "update submodule ",
        "publishing web:",
        "no public description",
    ]

    SKIP_TEXT = [
        "update dependency ",
    ]


if __name__ == "__main__":
    main(**parse_args())


