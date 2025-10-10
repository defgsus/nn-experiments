"""
Used for the "interest aggregation" blog post
"""
import os
import pickle
from pathlib import Path
import json
import datetime
import re
from typing import Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA, FactorAnalysis
from sklearn.cluster import Birch


class Calculator:

    def __init__(
            self,
            profiles_filename: Path = Path("datasets/porn-profiles-146k.json"),
            save_path: Path = Path("docs/posts/2025/"),
            cache_path : Path = Path("cache/porn-profiles"),
    ):
        self.save_path = save_path
        self.cache_path = cache_path
        self.min_interests_per_user = 5
        self.min_vertex_count = 30  #30
        self.min_edge_count = 0
        self.num_pca_components = 300
        self.profiles = json.loads(profiles_filename.read_text())
        print(f"Scraped profiles: {len(self.profiles):,}")

    def norm_name(self, name: str) -> Optional[str]:
        # fix other people's encoding problems
        if 0:
            name = (
                name
                .replace("Ã„", "Ö")
                .replace("Ã–", "Ö")
                .replace("Ã¶", "ö")
                .replace("Å‘", "ö")
                .replace("Ãœ", "Ü")
                .replace("Ã¼", "ü")
                .replace("ÃŸ", "ß")
                .replace("â€™", "'")
                .replace("Ã‰", "é")
                .replace("â™", "!")  # not sure about this
            )
            try:
                name.encode("latin1")
            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                print("ENCODING PROBLEM:", repr(name))
                return
        name = name.lower()
        #if "ss" not in name:
        #    name = name.rstrip("s")
        return name

    def get_graph(self):
        TODAY = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        RE_AGO = re.compile(r"(\d+)\s([a-z]+)[^\d]*")

        def ago_to_date(ago: str) -> datetime.datetime:
            try:
                if not isinstance(ago, str):
                    return ago
                if ago == "Now":
                    return TODAY
                kwargs = {}
                for amt, what in RE_AGO.findall(ago):
                    amt = int(amt)
                    if not what.endswith("s"):
                        what += "s"
                    if what == "months":
                        what = "days"
                        amt = amt * 30
                    if what == "years":
                        what = "days"
                        amt = amt * 365
                    kwargs[what] = kwargs.get(what, 0) + amt
                return TODAY - datetime.timedelta(**kwargs)
            except:
                print(repr(ago))
                raise

        vertex_map = {}
        edge_map = {}
        self.usernames = []
        for p in tqdm(self.profiles.values(), "parsing interests"):
            if not p.get("Signed up"):
                date = None
            else:
                date = ago_to_date(p["Signed up"]).strftime("%Y-%m-%d")
            interests = set(self.norm_name(n) for n in (p.get("Interests") or []))
            if None in interests:
                interests.remove(None)
            interests = sorted(interests)
            # store normalized version per profile
            p["interests"] = interests
            if len(interests) < self.min_interests_per_user:
                continue
            self.usernames.append(p["username"])
            for i, a in enumerate(interests):
                if a not in vertex_map:
                    vertex_map[a] = {"count": 0, "date": "2026-01-01"}
                vertex_map[a]["date"] = min(vertex_map[a]["date"], date)
                vertex_map[a]["count"] += 1
                for b in interests[i + 1:]:
                    key = tuple(sorted((a, b)))
                    edge_map[key] = edge_map.get(key, 0) + 1

        print(f"users with >={self.min_interests_per_user} interests: {len(self.usernames):,}")
        print(f"original: {len(vertex_map):,} interests x {len(edge_map):,} edges")
        vertex_map, edge_map = self.filter_graph(
            vertex_map, edge_map, min_vertex_count=self.min_vertex_count, min_edge_count=self.min_edge_count
        )
        print(f"filtered: {len(vertex_map):,} interests x {len(edge_map):,} edges")

        # sort them
        self.vertex_map = {
            key: vertex_map[key]
            for key in sorted(sorted(vertex_map), key=lambda key: -vertex_map[key]["count"])
        }
        self.edge_map = {
            key: edge_map[key]
            for key in sorted(sorted(edge_map), key=lambda key: -edge_map[key])
        }

        (self.cache_path / "profiles-cleaned.json").write_text(json.dumps(self.profiles))

    def filter_graph(self, vertex_map, edge_map, min_vertex_count: int = 30, min_edge_count: int = 0):
        print(f"filtering min_vertex_count={min_vertex_count}, min_edge_count={min_edge_count}")

        edge_map_f = {
            key: count
            for key, count in edge_map.items()
            if count >= min_edge_count
        }
        edge_vertices = set(k[0] for k in edge_map_f) | set(k[1] for k in edge_map_f)
        vertex_map_f = {
            n: v
            for n, v in vertex_map.items()
            if n in edge_vertices and v["count"] >= min_vertex_count
        }
        edge_map_f = {
            (a, b): count
            for (a, b), count in edge_map.items()
            if a in vertex_map_f and b in vertex_map_f
        }
        return vertex_map_f, edge_map_f

    def run(self):
        self.get_graph()
        self.calc_pca()

        vertex_positions = []
        for n_comp in sorted({10, 50, 100, 200, self.num_pca_components}):
            if n_comp <= self.num_pca_components:
                vertex_positions.append(
                    (f"pca{n_comp}-tsne2", 0, n_comp)
                )
        for n_comp in (50, 100, 200, 250):
            vertex_positions.append(
                (f"pca[{n_comp}:{self.num_pca_components}]-tsne2", n_comp, None)
            )
        vertex_positions = [
            {
                "name": name,
                "data": [
                    [round(f, 4) for f in row]
                    for row in self.calc_tsne(comp_min=comp_min, comp_max=comp_max).tolist()
                ]
            }
            for name, comp_min, comp_max in vertex_positions
        ]
        vertex_positions.insert(0, {
            "name": "pca2",
            "data": [
                [round(f, 4) for f in row]
                for row in self.interests_pca[:, :2].tolist()
            ]
        })

        filename = self.save_path / "interest-graph.json"
        vertex_to_id = {v: i for i, v in enumerate(self.vertex_map.keys())}
        size = filename.write_text(json.dumps({
            "limits": {
                "min_interests_per_user": self.min_interests_per_user,
                "min_interest_count": self.min_vertex_count,
                "min_interest_edge_count": self.min_edge_count,
                "pca_components": self.num_pca_components,
            },
            "num_vertices": len(self.vertex_map),
            "num_edges": len(self.edge_map),
            "num_users": len(self.usernames),
            "vertices": list(self.vertex_map.keys()),
            "edges": [[vertex_to_id[a], vertex_to_id[b], c] for (a, b), c in self.edge_map.items()],
            "vertex_counts": [v["count"] for v in self.vertex_map.values()],
            "vertex_positions": vertex_positions,
            # "vertex_pca_features": self.interests_pca.round(4).tolist(),
        }, separators=(",", ":")))
        print(f"Saved {size:,} bytes in {filename}")

        os.makedirs(self.cache_path, exist_ok=True)
        with (self.cache_path / "pca.pkl").open("wb") as fp:
            pickle.dump(self.pca, fp)
        np.savez(self.cache_path / "vertex_pca_fetures.npz", self.interests_pca)
        (self.cache_path / "usernames.json").write_text(json.dumps(self.usernames))

    def calc_pca(self):
        interests = list(self.vertex_map)

        def iter_matrix_batches():
            batch_size = self.num_pca_components
            for batch_idx in tqdm(range(0, len(interests), batch_size), desc=f"calc PCA{self.num_pca_components}"):
                batch_interests = interests[batch_idx: batch_idx + batch_size]
                batch_interest_ids = {n: i for i, n in enumerate(batch_interests)}
                batch = np.zeros((len(batch_interests), len(self.usernames)))
                for u_i, u in enumerate(self.usernames):
                    for i in self.profiles[u]["interests"]:
                        if i in batch_interest_ids:
                            batch[batch_interest_ids[i], u_i] = 1
                yield batch

        pca = IncrementalPCA(self.num_pca_components)
        for batch in iter_matrix_batches():
            pca.partial_fit(batch)

        interests_pca = []
        for batch in iter_matrix_batches():
            interests_pca.append(pca.transform(batch))
        self.interests_pca = np.concat(interests_pca)
        self.pca = pca

    def calc_tsne(
            self,
            comp_min: Optional[int] = None,
            comp_max: Optional[int] = None,
            dimensions: int = 2,
    ):
        tsne = TSNE(n_components=dimensions, verbose=1)
        vertex_pos: np.ndarray = tsne.fit_transform(self.interests_pca[:, comp_min: comp_max])
        return vertex_pos


def main():
    calculator = Calculator()
    calculator.run()


if __name__ == "__main__":
    main()
