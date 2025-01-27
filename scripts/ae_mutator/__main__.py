import argparse
import asyncio
import io
import json
from pathlib import Path

import tornado
import torchvision.transforms.functional as VF

from scripts.ae_mutator.mutator import Mutator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_file", type=str,
        help="The experiment.yml file to load the model and dataset"
    )

    return vars(parser.parse_args())


def run_server(
        experiment_file: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        device: str = "cpu",
):
    mutator = Mutator(
        experiment_file=experiment_file,
        device=device,
    )

    # deliver the html/js page
    class IndexHandler(tornado.web.RequestHandler):
        def get(self):
            self.write(
                (Path(__file__).resolve().parent / "index.html").read_text()
            )

    class ScriptHandler(tornado.web.RequestHandler):
        def get(self):
            self.write(
                (Path(__file__).resolve().parent / "main.js").read_text()
            )

    class MutatorHandler(tornado.web.RequestHandler):
        def get(self):
            self.write_data()

        def post(self):
            data = json.loads(self.request.body)
            if data["action"] == "mutate":
                mutator.push()
                mutator.mutate(data["x"], data["y"], amount=data["amount"])
            elif data["action"] == "undo":
                mutator.pop()

            self.write_data()

        def write_data(self):
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({
                "resolution": mutator.images.shape[-2:],
                "images": [
                    [
                        {
                            "x": x,
                            "y": y,
                            "filename": f"image-{image_id}.png"
                        }
                        for x, image_id in enumerate(row)
                    ]
                    for y, row in enumerate(mutator.image_ids)
                ],
            }))

    class ImageHandler(tornado.web.RequestHandler):
        def get(self, number: str):
            number = int(number)
            idx = None
            for y, row in enumerate(mutator.image_ids):
                for x, image_id in enumerate(row):
                    if image_id == number:
                        idx = y * mutator.width + x
                        break

            if idx is None:
                self.set_status(404)
                self.write("image not found")
            else:
                image = mutator.images[idx]
                image = VF.to_pil_image(image)
                fp = io.BytesIO()
                image.save(fp, format="png")
                fp.seek(0)
                self.set_header("Content-Type", "image/png")
                self.write(fp.read())

    async def main():
        application = tornado.web.Application([
            (r"/", IndexHandler),
            (r"/main.js", ScriptHandler),
            (r"/mutator/", MutatorHandler),
            (r"/image-(\d+).png", ImageHandler),
        ])
        print(f"visit http://{host}:{port}")
        application.listen(port=port, address=host)
        await asyncio.Event().wait()

    asyncio.run(main())


if __name__ == "__main__":
    run_server(**parse_args())