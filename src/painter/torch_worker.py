import multiprocessing
import queue
import time
import traceback
import uuid
from typing import Optional, Generator

from torch.multiprocessing import Process, Queue

from .torch_task import TorchTask


class TorchWorker:

    def __init__(self, verbose: bool = True):
        self._verbose = verbose
        self._process: Optional[Process] = None
        self._task: Optional[TorchTask] = None
        self._queue_in = Queue()
        self._queue_out = Queue()

    def running(self) -> bool:
        return bool(self._process)

    def start(self):
        if self.running():
            return
        if self._verbose:
            print("starting worker")

        multiprocessing.get_context("spawn")
        self._process = Process(target=self._main_loop)
        self._process.start()
        if self._verbose:
            print("worker started")

    def stop(self, join_queue: bool = True):
        if not self.running():
            return

        if self._verbose:
            print("stopping worker")

        self.put({"stop": True})

        self._queue_in.close()
        if join_queue:
            self._queue_in.join_thread()

        self._process.join()
        self._process = None

        if self._verbose:
            print("worker stopped")

    def put(self, data: dict):
        self._queue_in.put_nowait(data)

    def call(self, command: str, *args, **kwargs) -> str:
        """
        Call TorchTask.<command>(*args, **kwargs)
        returns: str, uuid of the call request
        """
        id = str(uuid.uuid4())
        if self._verbose:
            print(f"worker.call('{command}', '{id}')")
        self.put({"call": {"command": command, "args": args, "kwargs": kwargs, "uuid": id}})
        return id

    def poll(self, blocking: bool = False) -> Generator[dict, None, None]:
        while True:
            try:
                event =self._queue_out.get(block=blocking)
                if self._verbose:
                    print(f"worker.poll: {event.keys()}")
                yield event

            except queue.Empty:
                if not blocking:
                    break

    def _main_loop(self):
        try:
            self._task = TorchTask()
            self._queue_out.put_nowait({"info": self._task.info()})
            while True:
                try:
                    action = self._queue_in.get(timeout=1. / 1000.)

                    if action.get("stop"):
                        break

                    if action.get("call"):
                        try:
                            result = {
                                "result": getattr(self._task, action["call"]["command"])(*action["call"]["args"], **action["call"]["kwargs"])
                            }
                        except Exception as e:
                            result = {
                                "exception": {
                                    "type": type(e).__name__,
                                    "message": repr(e),
                                    "traceback": traceback.format_exc(),
                                }
                            }
                        self._queue_out.put({"call": {"uuid": action["call"]["uuid"], **result}})

                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            pass
