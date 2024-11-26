import pygame

from .image import Image
from .painter import Painter, PaintSequence
from .torch_worker import TorchWorker


class Application:

    def __init__(self):
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.running = False

        self.screen.fill("purple")

        self.image = Image.from_file(
            #"/home/bergi/Pictures/__diverse/02capitalism-tm.jpg"
            "/home/bergi/Pictures/__diverse/shenandoah2.jpg"
        )
        self.image_surface = self.image.to_pygame_surface()
        self.worker = TorchWorker()
        self.worker.start()
        self.painter = Painter(worker=self.worker)

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

        # flip() the display to put your work on screen
        pygame.display.flip()

    def mouse_move(self, event: pygame.event.Event):
        if event.dict["buttons"][0]:
            self.painter.paint(event.dict["pos"][0], event.dict["pos"][1])

    def mouse_down(self, event: pygame.event.Event):
        pass

    def mouse_up(self, event: pygame.event.Event):
        pass
