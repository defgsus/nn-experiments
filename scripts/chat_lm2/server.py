import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional, Literal

from websockets.asyncio.server import broadcast, serve, ServerConnection

from scripts.chat_lm2.chat import ChatModel

LOG_PATH = Path(__file__).resolve().parent / "logs"


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save-output", type=str, default=None,
        help=f"Store all chats in {LOG_PATH / '<save>'}",
    )
    parser.add_argument(
        "-b", "--bits", type=int, default=None,
        help=f"Weight quantization",
    )

    return vars(parser.parse_args())



class Server:

    def __init__(
            self,
            save_output: Optional[str] = None,
            bits: Optional[int] = None,
    ):
        self.sessions = {}
        self.model = ChatModel(
            log_path=None if save_output is None else (LOG_PATH / save_output),
            bits=bits,
        )

    def run(self):
        asyncio.run(self.main_loop())

    def info(self) -> dict:
        return {
            "model": self.model.model_name,
            "bits": self.model.bits,
            "tokens_per_sec": round(self.model.tokens_per_second, 2),
            "num_sessions": len(self.sessions),
        }

    async def send(self, websocket, data: dict):
        print("OUT", data)
        await websocket.send(json.dumps(data))

    async def send_all(self, data: dict):
        broadcast(self.sessions.keys(), json.dumps(data))

    def construct_message(
            self,
            type: Literal["info", "error", "completion", "last_prompts"],
            data: dict,
    ):
        return {"type": type, "data": data}

    async def handler(self, websocket: ServerConnection):
        try:
            self.sessions[websocket] = {}

            await self.send_all(self.construct_message("info", self.info()))

            async for message in websocket:
                await self.handle_message(websocket, json.loads(message))
                #await self.send(websocket, {"back": event})

        finally:
            self.sessions.pop(websocket)

    async def main_loop(self):
        async with serve(self.handler, "localhost", 6789) as server:
            await server.serve_forever()

    async def handle_message(self, websocket: ServerConnection, message: dict):
        print("IN", message)
        try:
            if command := message.get("command"):
                if func := getattr(self, f"command_{command}", None):
                   await func(websocket=websocket, **(message.get("kwargs") or {}))
        except Exception as e:
            await self.send(websocket, self.construct_message("error", {"message": f"{type(e).__name__}: {e}"}))

    async def command_hello(self, websocket: ServerConnection):#
        await self.send(websocket, self.construct_message("info", self.info()))
        await self.send(websocket, self.construct_message("last_prompts", self.model.last_stored_prompts()))

    async def command_generate(
            self,
            websocket: ServerConnection,
            blocks: List[dict],
            temperature: float = 1.,
            do_sample: Optional[bool] = None,
    ):
        last_info_time = 0

        message = None
        try:
            for text in self.model.generate(
                    blocks=blocks,
                    temperature=temperature,
                    do_sample=do_sample,
            ):
                cur_time = time.time()
                if cur_time - last_info_time >= 1.:
                    await self.send(websocket, self.construct_message("info", self.info()))
                    last_info_time = cur_time

                if text:
                    await self.send(websocket, self.construct_message("completion", {"text": text}))

                # break on new messages
                try:
                    async with asyncio.timeout(1/10):
                        message = await websocket.recv()
                        print("BREAK")
                        break
                except TimeoutError:
                    pass
        except StopIteration:
            pass

        await self.send(websocket, self.construct_message("completion", {"end": True}))

        if message:
            await self.handle_message(websocket, json.loads(message))


if __name__ == "__main__":
    Server(**parse_args()).run()
