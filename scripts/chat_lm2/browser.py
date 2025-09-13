import argparse
import asyncio
from pathlib import Path

import tornado
import tornado.web


def parse_args():
    parser = argparse.ArgumentParser()

    return vars(parser.parse_args())


def run_server(
        host: str = "127.0.0.1",
        port: int = 8000,
        **kwargs,
):
    # deliver the html/js page
    class MainHandler(tornado.web.RequestHandler):
        def get(self):
            self.write(
                (Path(__file__).resolve().parent / "browser.html")
                .read_text()
                .replace("__WEBSOCKET_URL__", "localhost:6789")
            )

    async def main():
        application = tornado.web.Application([
            (r"/", MainHandler),
        ])
        print(f"visit http://{host}:{port}")
        application.listen(port=port, address=host)
        await asyncio.Event().wait()

    asyncio.run(main())


if __name__ == "__main__":
    run_server(**parse_args())
