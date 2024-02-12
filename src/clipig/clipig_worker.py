import queue
import traceback
from copy import deepcopy
import threading
from typing import Tuple, Optional, Dict, Hashable, Generator, List, Iterable

import torch
import torch.nn as nn


from .clipig_task import ClipigTask
from ..models.clip import ClipSingleton
from ..util import to_torch_device


class ClipigWorker:

    def __init__(
            self,
            verbose: bool = False,
    ):
        self.verbose = verbose
        self._queue_in = queue.Queue()
        self._queue_out = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._task_map: Dict[Hashable, Tuple[ClipigTask, Generator]] = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def running(self) -> bool:
        return bool(self._thread)

    def start(self):
        if self.running():
            return

        self._thread = threading.Thread(name="clipig_worker", target=self._main_loop)
        self._thread.start()

    def stop(self, join_queue: bool = True):
        if not self.running():
            return

        self._queue_in.put({"stop": True})

        if join_queue:
            self._queue_in.join()

        self._thread.join()
        self._thread = None

    def run_task(self, id: Hashable, config: dict):
        self._queue_in.put({"run_task": id, "config": deepcopy(config)})

    def stop_task(self, id: Hashable):
        self._queue_in.put({"stop_task": id})

    def events(self, blocking: bool = False) -> Generator[dict, None, None]:
        while True:
            try:
                event = self._queue_out.get(block=blocking)
                if self.verbose:
                    print("EVENT:", event)
                yield event

            except queue.Empty:
                if not blocking:
                    break

    def _main_loop(self):
        try:
            while True:
                try:
                    action = self._queue_in.get(timeout=1. / 1000.)

                    self._queue_in.task_done()

                    if action.get("stop"):
                        break

                    if action.get("stop_task"):
                        self._stop_task(task_id=action["stop_task"])

                    if action.get("run_task"):
                        self._start_task(task_id=action["run_task"], config=action["config"])

                except queue.Empty:
                    pass

                next_task_map = {}
                for task_id, (task, iterable) in self._task_map.items():
                    try:
                        message = next(iterable)
                        if list(message.keys()) == ["status"]:
                            self._queue_out.put({"task": {"id": task_id, "status": message["status"]}})
                        else:
                            self._queue_out.put({"task": {"id": task_id, "message": message}})
                        next_task_map[task_id] = (task, iterable)

                    except StopIteration:
                        self._queue_out.put({"task": {"id": task_id, "status": "finished"}})

                    except Exception as e:
                        print(f"task '{task_id}' crashed with: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                        self._queue_out.put(
                            {"task": {"id": task_id, "status": "crashed", "exception": f"{type(e).__name__}: {e}"}}
                        )

                self._task_map = next_task_map

        except KeyboardInterrupt:
            pass

    def _start_task(self, task_id: Hashable, config: dict):
        if task_id in self._task_map:
            self._stop_task(task_id)

        task = ClipigTask(config=config)
        self._task_map[task_id] = (task, iter(task.run()))

        self._queue_out.put({"task": {"id": task_id, "status": "requested"}})

    def _stop_task(self, task_id: Hashable):
        if task_id in self._task_map:
            task, iterable = self._task_map.pop(task_id)
            del iterable
            del task

            self._queue_out.put({"task": {"id": task_id, "status": "stopped"}})
