import asyncio
import json
import logging
from typing import Optional, Generator, List

from websockets.sync.client import connect, Connection

# logging.basicConfig()

class Client:

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6789,
    ):
        self.host = host
        self.port = port
        self._websocket: Optional[Connection] = None
        self._blocks: List[dict] = []

    @property
    def websocket(self) -> Connection:
        if not self._websocket:
            self.connect()
        return self._websocket

    def connect(self):
        self._websocket = connect(f"ws://{self.host}:{self.port}")

    def close(self):
        if self._websocket:
            self._websocket.close()
            self._websocket = None

    def _send(self, data: dict):
        self.websocket.send(json.dumps(data))

    def send_command(self, command: str, **kwargs):
        self._send({"command": command, "kwargs": kwargs})

    def reset(self):
        # self.send_command("reset")
        self._blocks.clear()

    def add_block(self, role: str, content: str = ""):
        self._blocks.append({"role": role, "content": content})

    def generate(self) -> Generator[str, None, None]:
        self.send_command(
            "generate",
            blocks=self._blocks,
            temperature=1.01,
        )
        try:
            for message in self.websocket:
                message = json.loads(message)
                m_type, m_data = message["type"], message["data"]
                if m_type == "completion":
                    if text := m_data.get("text"):
                        yield text
                        if not self._blocks or self._blocks[-1]["role"] != "assistant":
                            self.add_block("assistant")
                        self._blocks[-1]["content"] += text
                    if m_data.get("end"):
                        break
        except Exception as e:
            yield f"\n{type(e).__name__}: {e}"


def main():
    client = Client()
    try:
        client.add_block("user", "Whus up?")

        gen = iter(client.generate())
        while True:
            print(next(gen))
        return
        for msg in client.generate():
            print(msg)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
