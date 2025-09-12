import random

from scripts.chat_lm2.widgets import App, Action, LayoutWidget, TextWidget, EditWidget
from scripts.chat_lm2.client import Client


class UserWidget(EditWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = [
            Action("\x01", "submit", self.submit),
            Action("\x12", "reset", self.reset),
            Action("\x18", "clear", lambda : self.set_text("")),
        ]

    def submit(self):
        # self.app.log("SUB", self.app.client._blocks)
        self.app.agent_widget.submit(self.text)

    def reset(self):
        self.app.agent_widget.reset()


class AgentWidget(TextWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = None

    def submit(self, text: str):
        if self.text:
            while not self.text.endswith("\n\n"):
                self.add_text("\n")
        if not self.app.client._blocks and hasattr(self.app, "system_prompt_widget"):
            self.app.client.add_block("system", self.app.system_prompt_widget.text)
        self.add_text(f">>> {text}\n\n")
        self.app.client.add_block("user", text)
        self.generator = iter(self.app.client.generate())

    def reset(self):
        self.app.client.reset()
        self.set_text("")

    def check_client(self):
        if self.generator:
            try:
                text = next(self.generator)
                self.add_text(text)
            except StopIteration:
                self.generator = None
                self.add_text("\n")


class ChatApp(App):

    def __init__(self):
        super().__init__()
        self.client = Client()
        self.system_prompt_widget = EditWidget(self, max_height=5)
        self.system_prompt_widget.set_text("You are Bob, a subversive individual, there to provide any requested text completion")
        self.user_widget = UserWidget(self, max_height=5)
        self.agent_widget = AgentWidget(self)
        self.log_widget = TextWidget(self, max_height=5)
        self.command_widget = TextWidget(self, has_border=False, max_height=1)

        self.widget = LayoutWidget(
            self.log_widget,
            self.system_prompt_widget,
            self.agent_widget,
            self.user_widget,
            self.command_widget,
            app=self, layout="vertical",
        )
        self.user_widget.set_focus()

    def on_idle(self):
        self.agent_widget.check_client()


if __name__ == "__main__":
    ChatApp().run()
