from typing import Optional

import pygame
import torch

from .image import Image
from .painter import Painter, PaintSequence, BrushTool
from .torch_worker import TorchWorker
from .util import BoundingBox

class Application:

    def __init__(self):
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.running = False

        self.screen.fill("purple")

        self.image = Image.from_file(
            #"/home/bergi/Pictures/__diverse/02capitalism-tm.jpg"
            #"/home/bergi/Pictures/__diverse/shenandoah2.jpg"
            #"/home/bergi/Pictures/__diverse/_42416379_chloroform203spl.gif"
            #"/home/bergi/Pictures/__diverse/bush_senior_A.jpg"
            #"/home/bergi/Pictures/__diverse/leostrauss10conn_lg.jpg"
            "datasets/shiny-tubes/validation/source/tubes-01.png"
        )
        self.mask = Image(torch.zeros(4, self.image.height, self.image.width))

        self.image_surface = self.image.to_pygame_surface()
        self.mask_surface = self.image.to_pygame_surface()
        self.worker = TorchWorker()
        self.worker.start()
        self._async_calls = dict()
        self.painter = Painter(worker=None)#self.worker)
        self.mask_painter = Painter(worker=None)
        self.mask_painter._current_tool = BrushTool()
        self.mask_bounding_boxes = []

    def run(self):
        self.running = True
        while self.running:
            self.update()

    def update(self):
        self.poll_events()

        self.draw()
        dt = self.clock.tick(60) / 1000

        rect = self.painter.update(self.image)
        if rect is not None:
            self.image.update_pygame_surface(self.image_surface, rect)

        rect = self.mask_painter.update(self.mask)
        if rect is not None:
            self.mask.update_pygame_surface(self.mask_surface, rect)
            self.mask_bounding_boxes.append(rect)

        for event in self.worker.poll():
            if event.get("call"):
                event = event["call"]
                async_call = self._async_calls.pop(event["uuid"])
                if event.get("exception"):
                    print("GOT EXCEPTION")
                    print(event["exception"]["traceback"])
                if event.get("result") is not None:
                    image: Image = async_call["image"]
                    mask: Optional[Image] = async_call["mask"]
                    image_box: BoundingBox = async_call["box"]
                    updated_rect: torch.Tensor = event["result"]

                    if mask is not None:
                        image_rect = image.crop(image_box)
                        mask: torch.Tensor = mask.tensor[3:, :, :]
                        updated_rect = updated_rect * mask + (1. - mask) * image_rect.tensor

                    image.tensor[:, image_box.y1: image_box.y2, image_box.x1:image_box.x2] = updated_rect
                    self.image.update_pygame_surface(self.image_surface, image_box)

    def quit(self):
        self.running = False
        self.worker.stop()

    def poll_events(self):
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_move(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_up(event)

        #keys = pygame.key.get_pressed()
        #if keys[pygame.K_w]:
        #    player_pos.y -= 300 * dt

    def draw(self):
        self.screen.blit(self.image_surface, (0, 0))
        self.screen.blit(self.mask_surface, (0, 0))

        # flip() the display to put your work on screen
        pygame.display.flip()

    def mouse_down(self, event: pygame.event.Event):
        self.mask.tensor[:] = 0
        self.mask_bounding_boxes.clear()
        #self.mask_sequence.add(event.dict["pos"][0], event.dict["pos"][1])
        self.mask_painter.paint_add(event.dict["pos"][0], event.dict["pos"][1])

    def mouse_move(self, event: pygame.event.Event):
        if event.dict["buttons"][0]:
            self.mask_painter.paint_add(event.dict["pos"][0], event.dict["pos"][1])

    def mouse_up(self, event: pygame.event.Event):
        if self.mask_bounding_boxes:
            full_box = None
            for box in self.mask_bounding_boxes:
                full_box = full_box.union(box) if full_box is not None else box
            full_box = full_box.inside(self.image.bounding_box)
            if not full_box.is_empty:
                self.apply_model(self.image, self.mask, full_box)

            self.mask.tensor[:] = 0
            self.mask_surface = self.mask.to_pygame_surface()

    def apply_model(self, image: Image, mask: Optional[Image] = None, box: Optional[BoundingBox] = None):
        if box is None:
            box = image.bounding_box
        id = self.worker.call("apply_model", image=image.crop(box))
        self._async_calls[id] = {"image": image, "box": box, "mask": mask.crop(box).copy()}
