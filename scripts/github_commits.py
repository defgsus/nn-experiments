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
from typing import Optional, List

import pandas as pd
import torch
from tqdm import tqdm

from src.util import to_torch_device
from src.util.gharchive import GHArchive
from src.util.files import iter_ndjson, iter_lines
from src.util.embedding import normalize_embedding
from src.console import CC

BYTE_FREQ_FILTERS = {
    "text": torch.Tensor([
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.66160260e-02, 0.00000000e+00,
        0.00000000e+00, 1.54417180e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.87983900e-01, 7.20628770e-04, 3.70532330e-04, 8.61817900e-03,
        0.00000000e+00, 3.16037200e-05, 4.78569070e-04, 2.58333170e-03,
        8.13757700e-03, 8.30683950e-03, 1.93443550e-03, 1.47826570e-03,
        3.27952040e-03, 1.76277010e-02, 4.77110260e-02, 1.97479300e-02,
        1.17826610e-02, 1.38677180e-02, 1.54030390e-02, 5.49111560e-03,
        8.02700200e-03, 5.62606850e-03, 5.57729600e-03, 5.14340500e-03,
        5.70397600e-03, 5.56770200e-03, 2.11332960e-02, 6.07932800e-04,
        7.10109370e-04, 1.59122570e-03, 8.35538550e-04, 1.39822780e-03,
        6.76065600e-04, 2.81780570e-02, 7.74073300e-03, 1.42258590e-02,
        1.22465800e-02, 9.57588000e-03, 1.24583970e-02, 2.75082980e-03,
        2.42925040e-03, 7.54426750e-03, 1.78624910e-03, 3.07253800e-03,
        5.64063900e-03, 7.51844330e-03, 6.23054400e-03, 3.92627530e-03,
        9.41532900e-03, 1.12214960e-03, 7.99693700e-03, 1.18921390e-02,
        1.04251250e-02, 2.34024150e-02, 1.93630750e-03, 2.66554500e-03,
        5.26707150e-04, 8.41720670e-04, 2.27179300e-04, 6.27261350e-03,
        0.00000000e+00, 6.27261350e-03, 0.00000000e+00, 8.92988800e-03,
        2.72569850e-03, 2.14915680e-01, 4.00708950e-02, 1.16077620e-01,
        1.63148000e-01, 4.11897330e-01, 5.54672300e-02, 6.46963300e-02,
        5.58633060e-02, 1.91959020e-01, 2.32834260e-02, 2.77027430e-02,
        1.22344410e-01, 7.87444040e-02, 1.22028686e-01, 1.82735310e-01,
        1.38637700e-01, 7.39262000e-03, 1.94395960e-01, 5.42367700e-01,
        3.07779500e-01, 1.25801040e-01, 2.96294930e-02, 2.55989840e-02,
        3.67937270e-02, 4.08810500e-02, 2.85010500e-03, 7.89188500e-05,
        2.35616680e-04, 7.89188500e-05, 7.09904200e-04, 0.00000000e+00,
        3.36756350e-04, 1.16365270e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 6.93242240e-04, 5.73927130e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.36756350e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.35680390e-04,
        0.00000000e+00, 0.00000000e+00, 2.35680390e-04, 1.25429130e-04,
        3.36756350e-04, 2.35680390e-04, 0.00000000e+00, 0.00000000e+00,
        4.57561840e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.25429130e-04, 0.00000000e+00, 4.62185500e-04,
        4.91953000e-04, 0.00000000e+00, 0.00000000e+00, 4.41559620e-04,
        3.88852800e-04, 2.35680390e-04, 0.00000000e+00, 1.45253500e-04,
        4.18936920e-04, 4.09409640e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.57561840e-04, 0.00000000e+00, 0.00000000e+00,
        1.16365270e-04, 7.20748450e-04, 1.16365270e-04, 2.31013000e-04,
        1.16365270e-04, 1.16365270e-04, 2.35680390e-04, 0.00000000e+00,
        3.49095820e-04, 0.00000000e+00, 1.16365270e-04, 3.52045670e-04,
        2.35680390e-04, 3.36756350e-04, 1.16365270e-04, 3.36756350e-04,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.45784880e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.28001810e-03, 2.32730540e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.72436700e-04, 1.72324070e-03, 2.35680390e-04, 3.36756350e-04,
        0.00000000e+00, 3.36756350e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.25429130e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    ]),
}


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
        parser.add_argument(
            "--type", type=str, nargs="?", default="commit",
            choices=["commit", "comment"],
            help="The type of event to collect from gharchive"
        )

    def _add_text_filter_params(parser):
        parser.add_argument(
            "--min-text-length", type=int, nargs="?", default=0,
            help="consider texts with at least that number of characters",
        )
        parser.add_argument(
            "--bytefreq-filter", type=str, nargs="*", default=[],
            help="Apply a filter to the byte frequencies, e.g. `text/.9`",
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
            "--model", type=str, nargs="?", default="hug/thenlper/gte-small",
            help="The model used to encode text, can be `clip/<name>` (e.g. `clip/ViT-B/32`)"
                 " or a huggingface transformer"
                 ", like 'hug/thenlper/gte-small', 'sentence-transformers/all-MiniLM-L6-v2'"
                 ", check https://huggingface.co/spaces/mteb/leaderboard for a list of models sorted by size."
                 " Also accepts `bow/<length>`, `boc/<length>` and `bytes` for word/character histograms"
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


def get_filenames(path: str, date: datetime.date, type: str, min_text_length: int, model: str = ""):
    path = Path(path)
    mt = f"-tl{min_text_length}" if min_text_length > 0 else ""

    mn = model.replace('/', '-')

    class Filenames:
        commits = path / f"{type}s-{date}{mt}.ndjson.gz"
        info = path / f"{type}s-{date}{mt}.json"
        embeddings = path / f"{type}s-{date}{mt}-{mn}.b64.gz"
        html = Path(f"{type}s-{date}{mt}-{mn}.html")
    return Filenames


def command_show(
        path: str,
        date: str,
        days: int,
        type: str,
        min_text_length: int,
        bytefreq_filter: List[str],
):
    gharchive = GHArchive(verbose=True, raw_path=path)
    iterator = GHArchiveIterator(gharchive, bytefreq_filter)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for day in range(days):

        date = start_date + datetime.timedelta(days=day)
        filenames = get_filenames(path, date, type, min_text_length)

        print(f"creating file {filenames.commits} ...")
        os.makedirs(filenames.commits.parent, exist_ok=True)

        with gzip.open(filenames.commits, "wt") as fp:
            for obj in iterator.iter_type(type, start_date=date, days=1, min_text_length=min_text_length):
                print(obj)



def command_get(
        path: str,
        date: str,
        days: int,
        type: str,
        min_text_length: int,
        bytefreq_filter: List[str],
):
    gharchive = GHArchive(verbose=True, raw_path=path)
    iterator = GHArchiveIterator(gharchive, bytefreq_filter)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for day in range(days):

        date = start_date + datetime.timedelta(days=day)
        filenames = get_filenames(path, date, type, min_text_length)

        print(f"creating file {filenames.commits} ...")
        os.makedirs(filenames.commits.parent, exist_ok=True)

        total = 0
        abort = False
        try:
            with gzip.open(filenames.commits, "wt") as fp:
                for commit in iterator.iter_type(type, start_date=date, days=1, min_text_length=min_text_length):
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
        type: str,
        model: str,
        trust_remote_code: bool,
        min_text_length: int,
        bytefreq_filter: List[str],
        batch_size: int = 128,
):
    import torch
    from src.util.text_encoder import TextEncoder

    encoder = TextEncoder(model_name=model, trust_remote_code=trust_remote_code)

    start_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    try:
        for day in range(days):

            date = start_date + datetime.timedelta(days=day)
            filenames = get_filenames(path, date, type, min_text_length, model)

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
                        progress.desc = f"creating {model} embedding, date={commit['date']}"

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
        type: str,
        model: str,
        min_text_length: int,
        trust_remote_code: bool,
        bytefreq_filter: List[str],
):
    import numpy as np
    import torch
    from src.util.text_encoder import TextEncoder

    encoder = TextEncoder(model_name=model)

    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    commits, embeddings = load_data(path, date, type, min_text_length, model)

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

            def _repo_url(o: dict):
                return f"https://github.com/{o['repo']}"

            def _object_url(o: dict):
                if type == "commit":
                    return f"https://github.com/{o['repo']}/commit/{o['sha']}"
                elif type == "comment":
                    return f"https://github.com/{o['repo']}/issues/{o['issue_number']}#issue-{o['comment_id']}"
                return ""

            print()
            for idx in best_indices[-20:]:
                sim = similarities[idx]
                c = commits[idx]
                print(f"{CC.rgb(1., 1., 1.)}  {sim:.3f} {_repo_url(c)} {_object_url(c)}{CC.Off}")
                print(f"        {repr(c['message'][:2000])}")
            print()

    except KeyboardInterrupt:
        print()


def command_plot(
        path: str,
        date: str,
        type: str,
        model: str,
        trust_remote_code: bool,
        min_text_length: int,
        bytefreq_filter: List[str],
        limit: int,
):
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import plotly
    import plotly.express as px
    plotly.io.templates.default = "plotly_dark"

    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    filenames = get_filenames(path, date, type, min_text_length, model)
    commits, embeddings = load_data(path, date, type, min_text_length, model, limit)

    print(f"loaded embeddings: {embeddings.shape}")
    print(f"building tSNE...")

    reducer = TSNE(2, verbose=True, perplexity=20)
    positions = reducer.fit_transform(embeddings[:limit])

    print(f"clustering ...")
    clusterer = KMeans(50, n_init="auto")
    labels = clusterer.fit_predict(embeddings[:limit])

    data = {
        "x": positions[:limit, 0],
        "y": positions[:limit, 1],
        "repo": [c["repo"] for c in commits[:limit]],
        "message": [c["message"][:1000].replace("</script>", "") for c in commits[:limit]],
        "type": type,
    }
    if type == "commit":
        data.update({
            "sha": [c["sha"] for c in commits[:limit]],
        })
    elif type == "comment":
        data.update({
            "issue_number": [c["issue_number"] for c in commits[:limit]],
            "comment_id": [c["comment_id"] for c in commits[:limit]],
        })

    df = pd.DataFrame(data)
    fig = px.scatter(
        df,
        x="x", y="y",
        height=600,
        hover_data=["repo"],
        custom_data=[c for c in df.columns if c not in ("x", "y")],
        color=[str(c) for c in labels],
    )

    html = fig.to_html(
        full_html=False,
        include_mathjax=False,
        include_plotlyjs=True,
        post_script="""
            function commit_html(data) {
                let html = '';
                const repo_url = `<a href="https://github.com/${data[0]}" target="_blank">${data[0]}</a>`;
                const message = data[1];
                
                if (data[2] == "commit") {
                    const [repo, message, type, sha] = data;
                    html += `${repo_url}/commit/<a href="https://github.com/${repo}/commit/${sha}" target="_blank">${sha}</a>`;
                }
                else if (data[2] == "comment") {
                    const [repo, message, type, issue_number, comment_id] = data;
                    html += `${repo_url}/issues/<a href="https://github.com/${repo}/issues/${issue_number}#issue-${comment_id}" target="_blank">${issue_number}</a>`;
                }
                html = `<h4>${html}</h4>`;
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


def load_data(path: str, date: datetime.date, type: str, min_text_length: int, model: str, limit: Optional[int] = None):
    import numpy as np

    filenames = get_filenames(path, date, type, min_text_length, model)
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


class GHArchiveIterator:
    """
    Iterates through the gharchive commits
    and keeps a list commit messages to filter out duplicates.
    """
    def __init__(self, gharchive: GHArchive, byte_filters: List[str], message_buffer_size: int = 1_000_000):
        self.gharchive = gharchive
        self.message_buffer_size = message_buffer_size
        self.message_dict = {}
        self._message_dict_counter = 0
        self.byte_filters = None
        self.byte_filter_weights = None
        if byte_filters:
            self.byte_filters = torch.concat([
                BYTE_FREQ_FILTERS[f.split("/")[0]].unsqueeze(0)
                for f in byte_filters
            ])
            self.byte_filter_weights = [
                float(f.split("/")[1])
                for f in byte_filters
            ]

    def iter_type(
            self,
            type: str,
            start_date: datetime.date = datetime.date(2023, 11, 20),
            days: int = 7,
            min_text_length: int = 0,
    ):
        return getattr(self, f"iter_{type}s")(start_date, days, min_text_length)

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

                    if not self.is_message_accepted(message):
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

    def iter_comments(
            self,
            start_date: datetime.date = datetime.date(2023, 11, 20),
            days: int = 7,
            min_text_length: int = 0,
    ):
        iterables = [
            self.gharchive.iter_events(
                day=start_date + datetime.timedelta(i),
                event_type="IssueCommentEvent",
            )
            for i in range(days)
        ]
        iterable = itertools.chain(*iterables)

        num_skipped = 0
        num_yielded = 0

        with tqdm() as progress:
            for event in iterable:
                progress.update(1)
                progress.desc = (
                    f"messages/skips {num_yielded:,}/{num_skipped:,}"
                    f", buffer-size {len(self.message_dict):,}"
                    f", date={event['created_at']}"
                )

                num_skipped += 1
                user = event["payload"]["comment"]["user"]["login"]
                if user == "coveralls" or user.endswith("[bot]") or user.endswith("-bot"):
                    continue

                message = event["payload"]["comment"]["body"]
                if not self.is_message_accepted(message):
                    continue

                num_skipped -= 1
                self.add_to_message_dict(message)

                data = {
                    "repo": event["repo"]["name"],
                    "date": event["payload"]["comment"]["created_at"],
                    "issue_id": event["payload"]["issue"]["id"],
                    "issue_number": event["payload"]["issue"]["number"],
                    "issue_title": event["payload"]["issue"]["title"],
                    "labels": [l["name"] for l in event["payload"]["issue"]["labels"]],
                    #"issue_body": event["payload"]["issue"]["body"],
                    "comment_id": event["payload"]["comment"]["id"],
                    "user": event["payload"]["comment"]["user"]["login"],
                    "author_association": event["payload"]["comment"]["author_association"],
                    "message": message,
                }
                # print(json.dumps(event, indent=2))

                yield data
                num_yielded += 1

    def add_to_message_dict(self, message: str):
        self.message_dict[message.lower()] = self._message_dict_counter
        self._message_dict_counter += 1

        # shrink to ~half when too large
        if len(self.message_dict) >= self.message_buffer_size:
            median = sorted(self.message_dict.values())
            median = median[len(median) // 2]

            self.message_dict = {
                msg: step
                for msg, step in self.message_dict.items()
                if step <= median
            }

    def is_message_accepted(self, text: str) -> bool:
        if " " not in text:
            return False

        text_l = text.lower()
        if text_l in self.message_dict:
            return False

        for skip_start in self.SKIP_STARTS:
            if text_l.startswith(skip_start):
                return False

        for skip_text in self.SKIP_TEXT:
            if skip_text in text_l:
                return False

        for c in text_l:
            code = ord(c)
            if 0x300 <= code < 0x2000 or code >= 0x4000:
                return False

        if self.byte_filters is not None:
            from src.util.text_encoder import TextEncoder
            freqs = TextEncoder("bytefreq").encode([text])
            sim_matrix = freqs @ self.byte_filters.T
            for sim, weight in zip(sim_matrix, self.byte_filter_weights):
                if (weight >= 0 and sim < weight) or (weight < 0 and sim > -weight):
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
        "merges ",
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
        # issue comments
        "<",
        "## [codecov]",
        "[![",
        "* [![",
        "{\r",
        "{\n",
    ]

    SKIP_TEXT = [
        "update dependency ",
    ]


if __name__ == "__main__":
    main(**parse_args())


