import argparse
import asyncio
import json
import re
import threading
import time
import queue
from pathlib import Path

import torch
from IPython.terminal.prompts import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
from transformers.configuration_utils import PretrainedConfig

import tornado, tornado.websocket, tornado.web

from src.util import to_torch_device
from src.console import CC


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, nargs="?", default="blueapple8259/TinyStories-Alpaca",
    )
    parser.add_argument(
        "--device", type=str, nargs="?", default="cpu",
    )

    return vars(parser.parse_args())


class Chat:

    def __init__(
            self,
            model: str,
            device: str,

            temperature: float = 1.,
            max_length: int = 10_000,
            num_beams: int = 1,
    ):
        if model == "phi-2":
            model = "microsoft/phi-2"
        elif model == "phi-1_5":
            model = "microsoft/phi-1_5"
        self.model_name = model
        self.device = to_torch_device(device)
        self.temperature = temperature
        self.max_length = max_length
        self.num_beams = num_beams

        self.tokenizer_name = self.model_name
        if "tinystories" in self.model_name.lower():
            self.tokenizer_name = "EleutherAI/gpt-neo-125M"

        print(f"using device: {device}")
        print(f"loading model: {model}")

        torch.set_default_device(device)

        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)


class CustomTextStreamer(TextStreamer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_content = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_content.append(text)
        print(f"{CC.rgb(.6, .8, 1.)}{text}{CC.Off}", flush=True, end="" if not stream_end else None)


class FakeChat:

    def __init__(self):
        print("loading FakeChat...")
        time.sleep(3)
        print("FakeChat loaded")
        self._stop = False

    def iter_response(self, prompt: str):
        for c in prompt:
            time.sleep(.5)
            yield c


class ChatWorker:

    def __init__(self):
        self._thread = None
        self.chat = None
        self._queue = queue.Queue()

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

        self._do_stop = False
        self._thread = threading.Thread(name="chat_model", target=self._main_loop)
        self._thread.start()

    def stop(self, join_queue: bool = True):
        if not self.running() or self._do_stop:
            return

        self._queue.put({"stop": True})

        if join_queue:
            self._queue.join()

        self._do_stop = True
        self._thread.join()
        self._thread = None

    def prompt(self, prompt: str):
        self._queue.put({"prompt": prompt})

    def _main_loop(self):

        self.chat = FakeChat()
        response_iterable = None

        while not self._do_stop:
            try:
                action = self._queue.get(timeout=.01)

                if action.get("prompt"):
                    response_iterable = iter(self.chat.iter_response(prompt=action["prompt"]))

                if action.get("stop"):
                    self._do_stop = True

                else:
                    if response_iterable is not None:
                        try:
                            response = next(response_iterable)
                            self._queue.put({"response": response})

                        except StopIteration:
                            response_iterable = None

                self._queue.task_done()
            except queue.Empty:
                pass


def run_server(
        host: str = "127.0.0.1",
        port: int = 8000,
):
    ws_connection = None
    chat_worker = ChatWorker()
    chat_worker.start()

    class MainHandler(tornado.web.RequestHandler):
        def get(self):
            self.write(
                (Path(__file__).resolve().parent / "chat_lm_browser.html")
                .read_text()
                .replace("__WEBSOCKET_URL__", f"{host}:{port}/ws")
            )

    class WebSocketHandler(tornado.websocket.WebSocketHandler):
        def open(self):
            nonlocal ws_connection
            ws_connection = self.ws_connection
            print("WebSocket opened")
            self.write_message(json.dumps({"text": "Hi there!\n"}))

        def on_message(self, message):
            message = json.loads(message)
            print("INCOMING", message)
            if message.get("prompt"):
                chat_worker.prompt(message["prompt"])

        def on_close(self):
            nonlocal ws_connection
            ws_connection = None
            print("WebSocket closed")

    async def main():
        application = tornado.web.Application([
            (r"/", MainHandler),
            (r"/ws", WebSocketHandler),
        ])
        print(f"visit http://{host}:{port}")
        application.listen(port=port, address=host)
        await asyncio.Event().wait()

    try:
        asyncio.run(main())

    finally:
        chat_worker.stop()


if __name__ == "__main__":
    #chat(**parse_args())
    run_server()
