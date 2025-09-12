import asyncio
import json
from typing import List, Optional, Literal

from websockets.asyncio.server import broadcast, serve, Connection

from scripts.chat_lm2.chat import ChatModel


class Server:

    def __init__(self):
        self.sessions = {}
        self.model = ChatModel()

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

    async def handler(self, websocket: Connection):
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

    async def handle_message(self, websocket: Connection, message: dict):
        if command := message.get("command"):
            if func := getattr(self, f"command_{command}", None):
               await func(websocket=websocket, **message["kwargs"])

    async def command_generate(
            self,
            websocket: Connection,
            blocks: List[dict],
            temperature: float,
            do_sample: Optional[bool] = None,
    ):
        # hack: to make the polling client idle
        await self.send(websocket, self.construct_message("info", self.info()))

        for text in self.model.generate(
                blocks=blocks,
                temperature=temperature,
                do_sample=do_sample,
        ):
            await self.send(websocket, self.construct_message("completion", {"text": text}))
        await self.send(websocket, self.construct_message("completion", {"end": True}))


if __name__ == "__main__":
    Server().run()
