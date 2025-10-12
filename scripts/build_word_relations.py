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

import torch
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
        self.features: dict[str, torch.Tensor] = {}
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

    def get_cache_filename(self, name: str|Path) -> Path:
        sub_path = f"nv{len(self.vertex_map)}-nu{len(self.usernames)}" #-ne{len(self.edge_map)}"
        return self.cache_path / sub_path / name

    def run(self):
        self.get_graph()
        self.write_dataset()
        #self.calc_own_model_embeddings("pca-mlp6K", "experiments/ae/interest-graph-linear.yml")
        self.calc_sf_embeddings("granite107M")
        self.calc_sf_embeddings("zip1M", "tabularisai/Zip-1")
        self.calc_sf_embeddings("mdbr-mt22M", "MongoDB/mdbr-leaf-mt")
        self.calc_sf_embeddings("mdbr-ir22M", "MongoDB/mdbr-leaf-ir")
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
                    for row in self.calc_tsne("pca", comp_min=comp_min, comp_max=comp_max).tolist()
                ],
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
        vertex_positions.extend([
            {
                "name": f"{feature_name}-tsne2",
                "data": [
                    [round(f, 4) for f in row]
                    for row in self.calc_tsne(feature_name, source=self.features[feature_name]).tolist()
                ]
            }
            for feature_name in self.features
        ])

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

    def calc_own_model_embeddings(self, name: str, experiment_filename: str):
        from src.train.experiment import load_experiment_trainer

        cache_filename = self.get_cache_filename(f"features-own-{name}.npz")
        if cache_filename.exists():
            with cache_filename.open("rb") as fp:
                self.features[name] = np.load(fp)["arr_0"]
            return

        trainer = load_experiment_trainer(experiment_filename)
        trainer.model.eval()
        features = []
        for batch in self.iter_matrix_batches(32, desc=f"Calc {name}"):
            feature_batch = trainer.model.encode(torch.from_numpy(batch).to(trainer.device).to(torch.float32))
            features.append(feature_batch.detach().cpu())

        self.features[name] = torch.concat(features)
        with cache_filename.open("wb") as fp:
            np.savez(fp, self.features[name])

    def calc_sf_embeddings(
            self,
            name: str,
            model_path: str = "ibm-granite/granite-embedding-107m-multilingual",
    ):
        from transformers import AutoModel, AutoTokenizer

        cache_filename = self.get_cache_filename(f"features-{name}.npz")
        if cache_filename.exists():
            with cache_filename.open("rb") as fp:
                self.features[name] = np.load(fp)["arr_0"]
            return

        def iter_batches():
            batch = []
            for word in self.vertex_map.keys():
                batch.append(word)
                if len(batch) >= 100:
                    yield batch
                    batch.clear()
            if batch:
                yield batch

        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        with torch.no_grad():
            features_list = []
            for batch in tqdm(iter_batches(), desc=f"calc {name} embeddings"):
                tokens = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                features = model(**tokens)[0][:, 0]
                features_list.append(features)
        self.features[name] = torch.concat(features_list)
        with cache_filename.open("wb") as fp:
            np.savez(fp, self.features[name])

    def iter_matrix_batches(self, batch_size: int, desc: str|None = None):
        interests = list(self.vertex_map)
        for batch_idx in tqdm(range(0, len(interests), batch_size), desc=desc):
            batch_interests = interests[batch_idx: batch_idx + batch_size]
            batch_interest_ids = {n: i for i, n in enumerate(batch_interests)}
            batch = np.zeros((len(batch_interests), len(self.usernames)))
            for u_i, u in enumerate(self.usernames):
                for i in self.profiles[u]["interests"]:
                    if i in batch_interest_ids:
                        batch[batch_interest_ids[i], u_i] = 1
            yield batch

    def calc_pca(self):
        cache_filename = self.get_cache_filename(f"pca{self.num_pca_components}.pkl")
        if cache_filename.exists():
            with cache_filename.open("rb") as fp:
                self.pca = pickle.load(fp)
            with (cache_filename.parent / f"pca{self.num_pca_components}-features.npz").open("rb") as fp:
                self.interests_pca = np.load(fp)["arr_0"]
            return

        pca = IncrementalPCA(self.num_pca_components)
        for batch in self.iter_matrix_batches(batch_size=self.num_pca_components, desc=f"fit PCA{self.num_pca_components}"):
            pca.partial_fit(batch)

        interests_pca = []
        for batch in self.iter_matrix_batches(batch_size=self.num_pca_components, desc=f"calc PCA{self.num_pca_components}"):
            interests_pca.append(pca.transform(batch))
        self.interests_pca = np.concat(interests_pca)
        self.pca = pca
        os.makedirs(cache_filename.parent, exist_ok=True)
        with cache_filename.open("wb") as fp:
            pickle.dump(self.pca, fp)
        with (cache_filename.parent / f"pca{self.num_pca_components}-features.npz").open("wb") as fp:
            np.savez(fp, self.interests_pca)

    def calc_tsne(
            self,
            name: str,
            source: torch.Tensor|np.ndarray|None = None,
            comp_min: Optional[int] = None,
            comp_max: Optional[int] = None,
            dimensions: int = 2,
    ):
        cache_filename = self.get_cache_filename(f"tsne-{name}-{comp_min}-{comp_max}-{dimensions}.pkl")
        if cache_filename.exists():
            with cache_filename.open("rb") as fp:
                return np.load(fp)["arr_0"]

        tsne = TSNE(n_components=dimensions, verbose=1)
        if source is None:
            source = self.interests_pca[:, comp_min: comp_max]
        else:
            if hasattr(source, "to_numpy"):
                source = source.to_numpy()
        vertex_pos: np.ndarray = tsne.fit_transform(source)

        os.makedirs(cache_filename.parent, exist_ok=True)
        with cache_filename.open("wb") as fp:
            np.savez(fp, vertex_pos)
        return vertex_pos

    def write_dataset(self):
        cache_filename = self.get_cache_filename("interest-user-matrix.pt")
        if not cache_filename.exists():
            batches = []
            for batch in self.iter_matrix_batches(128, desc="store dataset"):
                batch = torch.from_numpy(batch.astype(np.uint8))
                batches.append(batch)
            batches = torch.concat(batches)
            torch.save(batches, cache_filename)


def main():
    calculator = Calculator()
    calculator.run()


if __name__ == "__main__":
    main()
