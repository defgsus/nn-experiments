import argparse
import asyncio
import json
import re
import threading
import time
import queue
from pathlib import Path
from typing import Callable

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
        "--model", type=str, nargs="?",
        #default="blueapple8259/TinyStories-Alpaca",
        default="microsoft/phi-1_5",
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

    def iter_response(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        while True:
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + 10,
                num_beams=self.num_beams,
                temperature=self.temperature,
                do_sample=self.temperature != 1,
            )
            output_ids = output_ids[:, input_ids.shape[-1]:]
            input_ids = torch.concat([input_ids, output_ids], dim=-1)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            yield output_text


class FakeChat:

    def __init__(self, **kwargs):
        print("loading FakeChat...")
        time.sleep(3)
        print("FakeChat loaded")
        self._stop = False

    def iter_response(self, prompt: str):
        for c in prompt:
            time.sleep(.5)
            yield c


class ChatWorker:

    def __init__(
            self,
            model: str,
            device: str,
    ):
        self._thread = None
        self.chat = None
        self._queue = queue.Queue()
        self.model = model
        self.device = device
        self._do_stop = False

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

    def prompt(self, prompt: str, callback: Callable):
        self._queue.put({"prompt": prompt, "callback": callback})

    def break_generation(self):
        self._queue.put({"break": True})

    def _main_loop(self):
        self.chat = Chat(model=self.model, device=self.device)
        try:
            self._main_loop_impl()
        except KeyboardInterrupt:
            pass

    def _main_loop_impl(self):
        response_iterable = None
        cur_callback = None

        while not self._do_stop:
            try:
                action = self._queue.get(timeout=.01)

                if action.get("prompt"):
                    response_iterable = iter(self.chat.iter_response(prompt=action["prompt"]))
                    cur_callback = action["callback"]

                if action.get("break"):
                    if cur_callback:
                        cur_callback({"message": "<break>"})
                    response_iterable = None
                    cur_callback = None

                self._queue.task_done()

                if action.get("stop"):
                    self._do_stop = True
                    break

            except queue.Empty:
                pass

            if response_iterable is not None:
                try:
                    response = next(response_iterable)
                    cur_callback({"text": response})

                except StopIteration:
                    response_iterable = None
                    cur_callback = None


def run_server(
        model: str,
        device: str,
        host: str = "127.0.0.1",
        port: int = 8000,
):
    state = {}

    def _callback(text: dict):
        if state.get("ioloop"):
            state["ioloop"].call_soon_threadsafe(
                lambda: state.get("ws") and state["ws"].write_message(json.dumps(text))
            )

    chat_worker = ChatWorker(model=model, device=device)
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
            state["ws"] = self.ws_connection
            state["ioloop"] = asyncio.get_event_loop()
            print("WebSocket opened")
            self.write_message(json.dumps({"message": f"Connected to model {model}\n"}))

        def on_message(self, message):
            message = json.loads(message)
            print("INCOMING", message)
            if message.get("prompt"):
                chat_worker.prompt(
                    prompt=message["prompt"],
                    callback=_callback,
                )
            if message.get("break"):
                chat_worker.break_generation()

        def on_close(self):
            state.pop("ws", None)
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
    run_server(**parse_args())
