import argparse
import re

import torch
from IPython.terminal.prompts import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
from transformers.configuration_utils import PretrainedConfig

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


def chat(
        model: str,
        device: str,

        temperature: float = 1.,
        max_length: int = 10_000,
        num_beams: int = 1,
        chat: bool = False,
):
    params = {
        "temperature": temperature,
        "max_length": max_length,
        "num_beams": num_beams,
        "chat": chat,
    }
    device = to_torch_device(device)

    if model == "phi-2":
        model = "microsoft/phi-2"
    elif model == "phi-1_5":
        model = "microsoft/phi-1_5"

    tokenizer_model = model
    if "tinystories" in model.lower():
        tokenizer_model = "EleutherAI/gpt-neo-125M"

    print(f"using device: {device}")
    print(f"loading model: {model}")

    torch.set_default_device(device)

    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True, local_files_only=True)

    print()
    chat_history = []
    try:
        while True:
            prompt = input(f"\n{CC.rgb(.6, .9, .7)}> ").replace("\\n", "\n")
            print(CC.Off)

            if prompt == "clear":
                chat_history.clear()
                continue

            if prompt.startswith("!"):
                try:
                    exec(prompt[1:], globals(), params)
                    print("X", params)
                except Exception as e:
                    print(f"{type(e).__name__}: {e}")

                continue

            if not params["chat"]:
                chat_history.clear()
            else:
                prompt = prompt + "\n"

            chat_history.append(prompt)
            input_prompt = "\n".join(chat_history)
            print(f"{CC.rgb(1., .7, .4)}{input_prompt}{CC.Off}")

            input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

            streamer = CustomTextStreamer(tokenizer)
            try:
                model.generate(
                    input_ids,
                    max_length=params["max_length"],
                    num_beams=params["num_beams"],
                    temperature=float(params["temperature"]),
                    do_sample=params["temperature"] != 1,
                    streamer=streamer,
                    no_repeat_ngram_size=50,
                )

            except KeyboardInterrupt:
                pass

            if params["chat"]:
                new_text = "".join(streamer.text_content)
                chat_history.append(new_text[len(input_prompt):])

    except KeyboardInterrupt:
        print("\nCiao")


class CustomTextStreamer(TextStreamer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_content = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_content.append(text)
        print(f"{CC.rgb(.6, .8, 1.)}{text}{CC.Off}", flush=True, end="" if not stream_end else None)


if __name__ == "__main__":
    chat(**parse_args())
