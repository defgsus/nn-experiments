import threading
import time
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from transformers.utils import get_json_schema


class ChatModel:

    def __init__(
            self,
            model_name: str = "ibm-granite/granite-3.3-2b-instruct",
            device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
            ).eval()
        return self._model

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
        config.temperature = temperature
        config.max_new_tokens = max_new_tokens

        class _Streamer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.output = []
                self.running = True

            def put(self, token_ids):
                if token_ids.shape[-1] > 1:
                    return
                token: str = self.tokenizer.batch_decode(token_ids)[0]
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

                    time.sleep(1/20)

                yield from self.output

        streamer = _Streamer(self.tokenizer)
        def _generate():
            try:
                self.model.generate(**input_tokens, streamer=streamer, generation_config=config)
            except StopIteration:
                pass

        thread = threading.Thread(target=_generate)
        thread.start()
        try:
            yield from streamer
        finally:
            if thread.is_alive():
                thread.join()


if __name__ == "__main__":
    for msg in ChatModel().generate([{"role": "user", "content": "Whas up?"}]):
        print("X", repr(msg))


