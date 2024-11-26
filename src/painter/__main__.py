import time

import torch
import pygame

from src.painter.app import Application
from src.painter.torch_worker import TorchWorker


def main():
    pygame.init()
    Application().run()
    pygame.quit()


def test_surface_speed():
    from tqdm import tqdm
    import torch
    from src.painter.image import Image
    image = Image(torch.rand((3, 1024, 1024)))
    for i in tqdm(range(1000)):
        surface = image.to_pygame_surface()


def test_worker():

    worker = TorchWorker()
    worker.start()

    worker.call("load_model", "experiments/diffusion/noisediffusion.yml")
    try:
        while True:
            print("checking events:")
            for event in worker.poll():
                print(f"event {event}")

            time.sleep(.3)

    except KeyboardInterrupt:
        pass

    worker.stop()



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    main()
    #test_surface_speed()
    #test_worker()