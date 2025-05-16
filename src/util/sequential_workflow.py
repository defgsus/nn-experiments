import shutil
import pickle
import sys
from pathlib import Path
from typing import Optional, Union, Any


class SequentialWorkflow:

    def __init__(
            self,
            root_path: Union[str, Path],
            reset: bool = False,
            verbose: bool = True,
    ):
        self._root_path = Path(root_path)
        self._verbose = verbose
        if reset:
            shutil.rmtree(self._root_path)

    def save_result(self, method_name: str, data: dict):
        filename = self._root_path / f"{method_name}.pkl"
        with filename.open("wb") as fp:
            pickle.dump(data, fp)

    def load_result(self, method_name: str) -> dict:
        filename = self._root_path / f"{method_name}.pkl"
        with filename.open("rb") as fp:
            return pickle.load(fp)

    def result_exists(self, method_name: str) -> bool:
        filename = self._root_path / f"{method_name}.pkl"
        return filename.exists() and filename.stat().st_size > 0

    def run(self):
        step_methods = sorted(
            (name, getattr(self, name))
            for name in dir(self)
            if name.startswith("step_") and callable(getattr(self, name))
        )

        if not step_methods:
            return None

        last_method_name: Optional[str] = None
        last_result: Optional[dict] = None
        for method_name, method in step_methods:

            if self.result_exists(method_name):
                if self._verbose:
                    print(f"{method_name}: skipping", file=sys.stderr, flush=True)
                last_method_name = method_name
                continue

            if self._verbose:
                print(f"{method_name}: executing", file=sys.stderr, flush=True)

            if last_method_name is not None:
                if last_result is None:
                    last_result = self.load_result(last_method_name)

            if last_result is None:
                last_result = method()
            else:
                last_result = method(**last_result)

            self.save_result(method_name, last_result)
            last_method_name = method_name

        if last_result is None:
            last_result = self.load_result(step_methods[-1][0])

        return last_result
