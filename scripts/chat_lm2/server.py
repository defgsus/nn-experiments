import argparse
import asyncio
import json
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

    return vars(parser.parse_args())



class Server:

    def __init__(
            self,
            save_output: Optional[str] = None,
    ):
        self.sessions = {}
        self.model = ChatModel(
            save_log_path=None if save_output is None else (LOG_PATH / save_output),
        )

    def run(self):
        asyncio.run(self.main_loop())

    def info(self) -> dict:
        return {
            "model": self.model.model_name,
            "num_sessions": len(self.sessions),
        }

    async def send(self, websocket, data: dict):
        await websocket.send(json.dumps(data))

    async def send_all(self, data: dict):
        broadcast(self.sessions.keys(), json.dumps(data))

    def construct_message(
            self,
            type: Literal["info", "completion"],
            data: dict,
    ):
        return {"type": type, "data": data}

    async def handler(self, websocket: ServerConnection):
        try:
            self.sessions[websocket] = {}

            #await self.send(websocket, self.info())
            await self.send_all(self.construct_message("info", self.info()))

            async for message in websocket:
                print("IN", message)
                await self.handle_message(websocket, json.loads(message))
                #await self.send(websocket, {"back": event})

        finally:
            self.sessions.pop(websocket)

    async def main_loop(self):
        async with serve(self.handler, "localhost", 6789) as server:
            await server.serve_forever()

    async def handle_message(self, websocket: ServerConnection, message: dict):
        if command := message.get("command"):
            if func := getattr(self, f"command_{command}", None):
               await func(websocket=websocket, **message["kwargs"])

    async def command_generate(
            self,
            websocket: ServerConnection,
            blocks: List[dict],
            temperature: float,
            do_sample: Optional[bool] = None,
    ):
        message = None
        for text in self.model.generate(
                blocks=blocks,
                temperature=temperature,
                do_sample=do_sample,
        ):
            if not text:
                await self.send(websocket, self.construct_message("info", self.info()))
            else:
                await self.send(websocket, self.construct_message("completion", {"text": text}))

            try:
                async with asyncio.timeout(.1):
                    message = await websocket.recv()
                    break
            except TimeoutError:
                pass

        await self.send(websocket, self.construct_message("completion", {"end": True}))

        if message:
            await self.handle_message(websocket, json.loads(message))


if __name__ == "__main__":
    Server(**parse_args()).run()
