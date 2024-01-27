import argparse
import asyncio
import json
import re
import threading
import time
import queue
from pathlib import Path
from typing import Callable, Optional

import torch
from IPython.terminal.prompts import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
from transformers.configuration_utils import PretrainedConfig

import tornado, tornado.websocket, tornado.web

from src.util import to_torch_device


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, nargs="?",
        #default="blueapple8259/TinyStories-Alpaca",
        default="microsoft/phi-1_5",
        help="One of the models listed on https://huggingface.co/models?pipeline_tag=text-generation",
    )
    parser.add_argument(
        "--device", type=str, nargs="?", default="cpu",
        help="A torch device name like 'cpu', 'cuda' or 'auto'"
    )
    parser.add_argument(
        "--bits", type=int, nargs="?", default=16,
        choices=[4, 8, 16],
        help="Weight precision reduction. See https://huggingface.co/docs/transformers/llm_tutorial_optimization#1-lower-precision"
    )
    parser.add_argument(
        "--checkpoint", type=str, nargs="?", default=None,
        help="Load a checkpoint from exp.py training"
    )

    return vars(parser.parse_args())


class Chat:
    """
    Wrapper around huggingface AutoModelForCausalLM that features iterating the generated tokens.
    """
    def __init__(
            self,
            model: str,
            device: str,
            bits: int = 16,
            checkpoint: Optional[str] = None,
    ):
        if model == "phi-2":
            model = "microsoft/phi-2"
        elif model == "phi-1_5":
            model = "microsoft/phi-1_5"
        self.model_name = model
        self.device = to_torch_device(device)

        self.tokenizer_name = self.model_name
        if "tinystories" in self.model_name.lower():
            self.tokenizer_name = "EleutherAI/gpt-neo-125M"

        print(f"using device: {device}")
        print(f"loading model: {model}")

        torch.set_default_device(device)

        print(f"loading model {model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            load_in_8bit=bits==8,
            load_in_4bit=bits==4,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)

        if checkpoint:
            print(f"loading {checkpoint}")
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint["state_dict"])

    def iter_response(self, prompt: str):
        # check example code at https://huggingface.co/docs/transformers/llm_tutorial_optimization#1-lower-precision
        next_id = self.tokenizer.encode(prompt, return_tensors="pt")
        past_key_values = None
        with torch.no_grad():
            while True:
                next_logits, past_key_values = self.model(
                    next_id, use_cache=True, past_key_values=past_key_values,
                ).to_tuple()
                next_logits = next_logits[:, -1:]

                next_id = next_logits.argmax(dim=-1)

                if next_id == self.tokenizer.eos_token_id:
                    break

                yield self.tokenizer.decode(next_id.item())

    # using model.generate() which is not really interactive..
    def iter_response_DEV(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        while True:
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + 10,
            )
            output_ids = output_ids[:, input_ids.shape[-1]:]
            input_ids = torch.concat([input_ids, output_ids], dim=-1)
            for id in output_ids[0]:
                yield self.tokenizer.decode(id, skip_special_tokens=True)


class FakeChat:
    """Just for testing the ChatWorker"""
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
    """
    Worker thread with chat model and simple queue.
    """
    def __init__(
            self,
            **kwargs,
    ):
        self._thread = None
        self.chat = None
        self._queue = queue.Queue()
        self.kwargs = kwargs
        self._do_stop = False
        self._last_token_time = None
        self._token_count = 0
        self._avg_tokens_per_sec = 0.

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
        """
        Put prompt to LM and call `callback` for each generated token.
        """
        self._queue.put({"prompt": prompt, "callback": callback})

    def break_generation(self):
        """
        Stop text completion and forget last callback
        """
        self._queue.put({"break": True})

    def _main_loop(self):
        self.chat = Chat(**self.kwargs)
        try:
            self._main_loop_impl()
        except KeyboardInterrupt:
            pass

    def _main_loop_impl(self):
        response_iterable = None
        cur_callback = None

        while not self._do_stop:
            try:
                action = self._queue.get(timeout=1./1000.)

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

                    cur_time = time.time()

                    self._token_count += 1
                    if self._last_token_time is None:
                        self._last_token_time = cur_time

                    time_passed = cur_time - self._last_token_time
                    if time_passed >= 1.:
                        tokens_per_sec = self._token_count / time_passed
                        self._avg_tokens_per_sec += .5 * (tokens_per_sec - self._avg_tokens_per_sec)
                        cur_callback({"status": {"tokens_per_sec": round(self._avg_tokens_per_sec, 2)}})
                        self._last_token_time = cur_time
                        self._token_count = 0

                except StopIteration:
                    response_iterable = None
                    cur_callback = None


def run_server(
        host: str = "127.0.0.1",
        port: int = 8000,
        **kwargs,
):
    state = {}
    kwargs["device"] = to_torch_device(kwargs["device"])

    # callback result from chat-worker is sent to websocket via the ioloop thread
    def _callback(data: dict):
        if state.get("ioloop"):
            state["ioloop"].call_soon_threadsafe(
                lambda: state.get("ws") and state["ws"].write_message(json.dumps(data))
            )

    chat_worker = ChatWorker(**kwargs)
    chat_worker.start()

    # deliver the html/js page
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
            self.write_message(json.dumps({"message": f"Connected to model {kwargs['model']} on {kwargs['device']}\n"}))

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
