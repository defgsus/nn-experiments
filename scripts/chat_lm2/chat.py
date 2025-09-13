import datetime
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from transformers.utils import get_json_schema


class ChatModel:

    def __init__(
            self,
            model_name: str = "ibm-granite/granite-3.3-2b-instruct",
            device: str = "cuda",
            bits: Optional[int] = None,
            log_path: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.bits = bits
        self.log_path = Path(log_path) if log_path is not None else None
        self._model = None
        self._tokenizer = None
        self.tokens_per_second: float = 0.

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            kwargs = {}
            if self.bits:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    **{f"load_in_{self.bits}bit": True},
                )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                **kwargs
            ).eval()
        return self._model

    def last_stored_chat_log(self) -> Optional[dict]:
        if self.log_path:
            filenames = sorted(self.log_path.glob("*.json"))
            if filenames:
                return json.loads(filenames[-1].read_text())

    def last_stored_prompts(self) -> Optional[dict]:
        if self.log_path:
            filenames = sorted(self.log_path.glob("*.json"), reverse=True)
            last_prompts = {}
            for fn in filenames:
                log = json.loads(fn.read_text())
                for block in log["blocks"]:
                    if block["role"] in ("system", "user") and block["content"]:
                        last_prompts[block["role"]] = block["content"]
                if len(last_prompts) == 2:
                    break
            return last_prompts

    def generate(
            self,
            blocks: List[dict],
            temperature: float = 1.,
            do_sample: Optional[bool] = None,
            max_new_tokens: int = 10_000,
    ):
        chat = self.tokenizer.apply_chat_template(blocks, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(chat, return_tensors="pt").to(self.device)
        config = GenerationConfig.from_model_config(self.model.config)
        config.do_sample = temperature != 1. if do_sample is None else do_sample
        config.temperature = None if not config.do_sample else temperature
        config.max_new_tokens = max_new_tokens

        class _Streamer:
            def __init__(self, parent: ChatModel):
                self.parent = parent
                self.output = []
                self.running = True
                self.last_token_time = None

            def put(self, token_ids):
                if not self.running:
                    raise StopIteration()

                # hacky: to ignore the intro text from the template
                if token_ids.shape[-1] > 1:
                    return

                if self.last_token_time is None:
                    self.last_token_time = time.time()
                else:
                    cur_time = time.time()
                    tokens_per_second = 1. / max(10e-9, cur_time - self.last_token_time)
                    self.parent.tokens_per_second += 1/8 * (tokens_per_second - self.parent.tokens_per_second)
                    self.last_token_time = cur_time

                token: str = self.parent.tokenizer.batch_decode(token_ids)[0]
                if token == "<|end_of_text|>":
                    self.running = False
                else:
                    self.output.append(token)

            def end(self):
                self.running = False

            def __iter__(self):
                while self.running:
                    while self.output:
                        yield self.output.pop(0)

                    yield ""
                    time.sleep(1/5)

                yield from self.output

        streamer = _Streamer(self)
        def _generate():
            try:
                self.model.generate(**input_tokens, streamer=streamer, generation_config=config)
            except StopIteration:
                pass

        thread = threading.Thread(target=_generate)
        thread.start()
        assistent_output = []
        try:
            for text in streamer:
                assistent_output.append(text)
                yield text
        finally:
            if thread.is_alive():
                streamer.running = False
                thread.join()

            if self.log_path:
                os.makedirs(self.log_path, exist_ok=True)
                (
                    self.log_path
                    / datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S.json")
                ).write_text(json.dumps({
                    "model": self.model_name,
                    "bits": self.bits,
                    "config": {
                        "do_sample": config.do_sample,
                        "temperature": config.temperature,
                        "max_new_tokens": config.max_new_tokens,
                    },
                    "blocks": [
                        *blocks,
                        {"role": "assistant", "content": "".join(assistent_output)},
                    ]
                }, indent=2))


if __name__ == "__main__":
    for msg in ChatModel(bits=4).generate([{"role": "user", "content": "Whas up?"}]):
        print("X", repr(msg))


