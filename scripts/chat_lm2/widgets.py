import curses
import dataclasses
import io
import time
import textwrap
from typing import Optional, List, Tuple, Literal, Dict, Callable


@dataclasses.dataclass
class Action:
    key: str
    name: str
    method: Callable[[], None]


class App:

    def __init__(self):
        self.window: Optional[curses.window] = None
        self.running: bool = False
        self.widget: Widget = EditWidget(self)
        self.log_widget: Optional[TextWidget] = None
        self.command_widget: Optional[TextWidget] = None
        self.fps: float = 30.
        self.time_delta: float = 1.
        self._last_refresh_time: float = time.time()
        self._last_render_time: float = .01
        self._window_size: Optional[Tuple[int, int]] = None
        self._focused_widget: Optional[Widget] = None
        self._curses_color_pairs: Dict[Tuple[int, int], int] = {}
        self.focus_next()

    def run(self):
        try:
            self.window = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.start_color()
            self.window.keypad(True)
            self.window.nodelay(True)
            self._run()
        finally:
            if self.window:
                self.window.keypad(False)
            curses.echo()
            curses.nocbreak()
            curses.endwin()

    def _run(self):
        self.running = True
        while self.running:

            try:
                key = self.window.getkey()
            except curses.error:
                key = None

            if key:
                self.on_key(key)

            cur_time = time.time()
            time_to_wait = 1. / self.fps - (cur_time - self._last_refresh_time) - self._last_render_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            window_size = self.window.getmaxyx()
            if self._window_size is None or self._window_size != window_size:
                self.on_resize(self._window_size, window_size)
                self._window_size = window_size

            cur_time = time.time()
            self.time_delta = cur_time - self._last_refresh_time
            self.on_idle()
            if self.time_delta >= 1. / self.fps:
                self.render()
                self._last_render_time = time.time() - cur_time
                self._last_refresh_time = cur_time

    def log(self, *args):
        if self.log_widget:
            fp = io.StringIO()
            print(*args, file=fp, end="")
            fp.seek(0)
            self.log_widget.set_text(f"{self.log_widget.text}\n{fp.read()}")

    def on_idle(self):
        pass

    def on_resize(self, old_size: Optional[Tuple[int, int]], new_size: Tuple[int, int]):
        self.widget.on_resize(new_size)

    def on_key(self, key: str) -> bool:
        if key in ("\t", "KEY_BTAB"):
            self.focus_next(backward=key == "KEY_BTAB")
            return True

        if w := self._focused_widget:
            while w:
                if w.on_action_key(key) or w.on_key(key):
                    return True
                w = w.parent
        else:
            if self.widget.on_action_key(key) or self.widget.on_key(key):
                return True

        return False

    def render(self):
        try:
            self.widget.render()
        except curses.error as e:
            e.add_note(f"in {self} for {self.widget}: screen_width/height={self._window_size[1]}/{self._window_size[0]}")
            raise

    def set_focused_widget(self, widget: "Widget"):
        if self._focused_widget:
            if self._focused_widget == widget:
                return
            self._focused_widget.set_dirty()
        self._focused_widget = widget
        self._focused_widget.set_dirty()
        if self.command_widget:
            self.command_widget.set_text(" ".join(
                f"{a.key}: {a.name}" for a in self._focused_widget.actions
            ))
    def focus_next(self, backward: bool = False):
        focusable_widgets = [w for w in self.iter_widgets() if w.is_focusable]
        if focusable_widgets:
            if not self._focused_widget:
                self.set_focused_widget(focusable_widgets[0])
            elif len(focusable_widgets) > 1:
                try:
                    idx = focusable_widgets.index(self._focused_widget)
                except IndexError:
                    return
                idx = (idx + (-1 if backward else 1)) % len(focusable_widgets)
                self.set_focused_widget(focusable_widgets[idx])

    def iter_widgets(self):
        def _iter(parent: Widget):
            yield parent
            for w in parent.children:
                yield from _iter(w)
        yield from _iter(self.widget)

    def get_curses_color(
            self,
            foreground: int,
            background: int = curses.COLOR_BLACK,
    ):
        key = (foreground, background)
        if key not in self._curses_color_pairs:
            idx = len(self._curses_color_pairs) + 1
            curses.init_pair(idx, foreground, background)
            self._curses_color_pairs[key] = curses.color_pair(idx)
        return self._curses_color_pairs[key]


class Widget:

    def __init__(
            self,
            *children: "Widget",
            app: App,
            x: int = 0,
            y: int = 0,
            width: int = 10,
            height: int = 10,
            max_width: Optional[int] = None,
            max_height: Optional[int] = None,
            is_focusable: bool = False,
            has_border: bool = False,
    ):
        self.app = app
        self.children: List[Widget] = list(children)
        for w in self.children:
            w.parent = self
        self.is_container: bool = bool(self.children)
        self.is_focusable = is_focusable
        self.has_border = has_border
        self._pos = (y, x)
        self._size = (height, width)
        self._max_size = (max_height, max_width)
        self._dirty: bool = True
        self.parent: Optional[Widget] = None
        self.actions: List[Action] = []

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self._pos[1]}, y={self._pos[0]}, width={self._size[1]}, height={self._size[0]})"

    def header_string(self) -> str:
        return f"p={self.pos} s={self.size}"

    def on_resize(self, size: Tuple[int, int]):
        for w in self.children:
            w.on_resize(size)

    def on_key(self, key: str) -> bool:
        return False

    def on_action_key(self, key: str) -> bool:
        for a in self.actions:
            if a.key == key:
                self.app.log("ACT", a)
                a.method()
                return True
        return False

    @property
    def window(self) -> curses.window:
        return self.app.window

    def set_dirty(self):
        self._dirty = True

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    def set_pos(self, pos: Tuple[int, int]):
        self._pos = pos
        self.set_dirty()

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def content_pos(self) -> Tuple[int, int]:
        return self._pos if not self.has_border else (self._pos[0] + 1, self._pos[1] + 1)

    @property
    def content_size(self) -> Tuple[int, int]:
        return self._size if not self.has_border else (self._size[0] - 2, self._size[1] - 2)

    def set_size(self, size: Tuple[int, int]):
        self._size = size
        self.set_dirty()

    def set_width(self, width: int):
        self._size = (self._size[0], width)
        self.set_dirty()

    def render(self):
        if self.has_border:
            self.render_border()

        for i, w in enumerate(sorted(self.children, key=lambda w: 1 if w.is_focus else 0)):
            try:
                w.render()
            except curses.error as e:
                e.add_note(f"in {self} for {i}. child {w}: screen_width/height={self.app._window_size[1]}/{self.app._window_size[0]}")
                raise

    @property
    def is_focus(self) -> bool:
        return self.app._focused_widget == self

    def set_focus(self):
        if self.app._focused_widget:
            if self.app._focused_widget == self:
                return
            self.app._focused_widget.set_dirty()
        self.app.set_focused_widget(self)

    def get_curses_color(self) -> int:
        color = self.app.get_curses_color(
            curses.COLOR_WHITE,
            curses.COLOR_BLACK,
            #curses.COLOR_GREEN if self.is_focus else curses.COLOR_BLACK,
        )
        if self.is_focus:
            color |= curses.A_BOLD
        return color

    def render_clear(self):
        line = " " * self.content_size[1]
        for y in range(self.content_size[0]):
            try:
                self.window.addstr(self.content_pos[0] + y, self.content_pos[1], line)
            except curses.error:
                pass

    def render_border(self):
        color = self.get_curses_color()
        self.window.addch(self._pos[0], self._pos[1], curses.ACS_ULCORNER, color)
        self.window.addch(self._pos[0], self._pos[1] + self._size[1] - 1, curses.ACS_URCORNER, color)
        self.window.addch(self._pos[0] + self._size[0] - 1, self._pos[1], curses.ACS_LLCORNER, color)
        try:
            self.window.addch(self._pos[0] + self._size[0] - 1, self._pos[1] + self._size[1] - 1, curses.ACS_LRCORNER, color)
        except curses.error:
            pass

        for x in range(self._pos[1] + 1, self._pos[1] + self._size[1] - 1):
            self.window.addch(self._pos[0], x, curses.ACS_HLINE, color)
            self.window.addch(self._pos[0] + self._size[0] - 1, x, curses.ACS_HLINE, color)

        for y in range(self._pos[0] + 1, self._pos[0] + self._size[0] - 1):
            self.window.addch(y, self._pos[1], curses.ACS_VLINE, color)
            self.window.addch(y, self._pos[1] + self._size[1] - 1, curses.ACS_VLINE, color)

        self.window.addnstr(self._pos[0], self._pos[1] + 2, self.header_string(), color, self._size[0] - 2)


class LayoutWidget(Widget):

    def __init__(
            self,
            *children: Widget,
            app: App,
            layout: Literal["horizontal", "vertical"] = "horizontal",
            x: int = 0,
            y: int = 0,
            width: int = 10,
            height: int = 10,
            max_width: Optional[int] = None,
            max_height: Optional[int] = None,
            has_border: bool = False,
    ):
        super().__init__(
            *children, app=app, x=x, y=y, width=width, height=height, max_width=max_width, max_height=max_height,
            has_border=has_border,
        )
        self._layout = layout

    def on_resize(self, size: Tuple[int, int]):
        self._size = (size[0] - self._pos[0], size[1] - self._pos[1])
        if not self.children:
            return

        if self._layout == "horizontal":
            widths = [1] * len(self.children)
            for _ in range(self.content_size[1]):
                for i, (width, w) in enumerate(zip(widths, self.children)):
                    if sum(widths) < self.content_size[1]:
                        if w._max_size[1] is None or width < w._max_size[1]:
                            widths[i] += 1
            x = 0
            for width, w in zip(widths, self.children):
                w.set_pos((self.content_pos[0], self.content_pos[1] + x))
                x += width
                size = (self.content_size[0], width)
                w.set_size(size)
                w.on_resize(size)
        else:
            heights = [1] * len(self.children)
            for _ in range(self.content_size[0]):
                for i, (height, w) in enumerate(zip(heights, self.children)):
                    if sum(heights) < self.content_size[0]:
                        if w._max_size[0] is None or height < w._max_size[0]:
                            heights[i] += 1
            y = 0
            for height, w in zip(heights, self.children):
                w.set_pos((self.content_pos[0] + y, self.content_pos[1]))
                y += height
                size = (height, self.content_size[1])
                w.set_size(size)
                w.on_resize(size)
        """
            w_width = self.content_size[1]
            w_height = self.content_size[0] // len(self.children)
            mul_y = w_height

        for i, w in enumerate(self.children):
            w.set_pos((self.content_pos[0] + i * mul_y, self.content_pos[1] + i * mul_x))
            if i == len(self.children) - 1:
                w_width = max(w_width, self.content_size[1] - w.pos[1])
                w_height = max(w_height, self.content_size[0] - w.pos[0])
            w.set_size((w_height, w_width))
            w.on_resize((w_height, w_width))
        """


class TextWidget(Widget):

    def __init__(
            self,
            app: App,
            x: int = 0,
            y: int = 0,
            width: int = 10,
            height: int = 10,
            max_width: Optional[int] = None,
            max_height: Optional[int] = None,
            is_focusable: bool = True,
            has_border: bool = True,
            auto_scroll: bool = True,
    ):
        super().__init__(
            app=app, x=x, y=y, width=width, height=height, max_width=max_width, max_height=max_height,
            is_focusable=is_focusable, has_border=has_border,
        )
        self.auto_scroll = auto_scroll
        self._text: str = ""
        self._text_lines: Optional[str] = None
        self._wrapped_text_lines: Optional[str] = None
        self._line_offset: int = 0
        self._cursor_pos: Tuple[int, int] = (0, 0)

    def header_string(self) -> str:
        return (
            f"{super().header_string()}"
            f" len={len(self.text)} lines={len(self.text_lines)} wraplines={len(self.wrapped_text_lines)}"
            f" lo={self._line_offset}"
        )

    @property
    def text(self):
        return self._text

    def set_text(self, text: str):
        self._text = text
        self._text_lines = None
        self._wrapped_text_lines = None
        self.set_dirty()

    def add_text(self, text: str):
        self.set_text(self.text + text)

    @property
    def text_lines(self):
        if self._text_lines is None:
            text_lines = self.text.splitlines()
            self._text_lines = []
            x = 0
            for line in text_lines:
                self._text_lines.append((x, line))
                x += len(line) + 1

        return self._text_lines

    @property
    def wrapped_text_lines(self):
        if self._wrapped_text_lines is None:
            self._wrapped_text_lines = []
            for line_idx, (_, line) in enumerate(self.text_lines):
                if not line:
                    self._wrapped_text_lines.append((line_idx, 0, ""))
                else:
                    wrapped_lines = textwrap.wrap(
                        line,
                        width=self.content_size[1],
                        break_long_words=True,
                        replace_whitespace=False,
                        expand_tabs=True, tabsize=4,
                        drop_whitespace=False,
                    )
                    wrapped_lines_with_index = []
                    x = 0
                    for w_line in wrapped_lines:
                        wrapped_lines_with_index.append((line_idx, x, w_line))
                        x += len(w_line)
                    self._wrapped_text_lines.extend(wrapped_lines_with_index)

        return self._wrapped_text_lines

    def end_of_wrapped_line(self, y: int) -> int:
        if 0 <= y < len(self.wrapped_text_lines):
            return len(self.wrapped_text_lines[y][-1])
        return 0

    def render(self):
        if self._dirty:
            self.render_clear()
            self._dirty = False

        if self.auto_scroll:
            self._line_offset = max(0, len(self.wrapped_text_lines) - self.content_size[0])

        if self.has_border:
            self.render_border()
        self.render_text()

    def render_text(self):
        lines = self.wrapped_text_lines
        for y, (_, _, line) in enumerate(lines[self._line_offset:][:self.content_size[0]]):
            self.window.addnstr(self.content_pos[0] + y, self.content_pos[1], line, self.content_size[1])

    def on_key(self, key: str) -> bool:
        if key == "KEY_UP":
            if self._line_offset > 0:
                self.auto_scroll = False
                self._line_offset -= 1
                self.set_dirty()
        elif key == "KEY_DOWN":
            if self._line_offset < len(self.wrapped_text_lines) - self.content_size[0]:
                self.auto_scroll = False
                self._line_offset += 1
            else:
                self.auto_scroll = True
            self.set_dirty()
        else:
            return False
        return True


class EditWidget(TextWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_scroll = False
        self.actions = [
            Action("\x18", "clear", lambda : self.set_text("")),
        ]

    def header_string(self) -> str:
        return (
            f"{super().header_string()} cur={self._cursor_pos} tpos={self.get_cursor_text_offset()}"
        )

    def on_key(self, key: str) -> bool:
        if key == "KEY_UP":
            if self._cursor_pos[0] > 0:
                y = self._cursor_pos[0] - 1
                self._cursor_pos = (y, min(self._cursor_pos[1], self.end_of_wrapped_line(y)))
                self.set_dirty()
        elif key == "KEY_DOWN":
            if self._cursor_pos[0] < len(self.wrapped_text_lines):
                y = self._cursor_pos[0] + 1
                self._cursor_pos = (y, min(self._cursor_pos[1], self.end_of_wrapped_line(y)))
                self.set_dirty()
        elif key == "KEY_LEFT":
            self.move_cursor(-1)
        elif key == "KEY_RIGHT":
            self.move_cursor(1)
        elif key == "KEY_BACKSPACE":
            ofs = self.get_cursor_text_offset()
            # self.app.log("ofs", ofs)
            self.move_cursor(-1)
            self.set_text(self.text[:ofs] + self.text[ofs + 1:])
        elif key == "kLFT5": # CTRL+LEFT
            self.set_cursor_text_offset(
                self._find_next_text_cursor_breakpoint(self.get_cursor_text_offset(), -1)
            )
        elif key == "kRIT5": # CTRL+RIGHT
            self.set_cursor_text_offset(
                self._find_next_text_cursor_breakpoint(self.get_cursor_text_offset(), +1)
            )
        elif key == "KEY_HOME":
            if self._cursor_pos[1] != 0:
                self.set_cursor_pos(self._cursor_pos[0], 0)
            else:
                self.set_cursor_pos(0, 0)
        elif key == "KEY_END":
            if self.wrapped_text_lines and 0 <= self._cursor_pos[0] < len(self.wrapped_text_lines):
                wl = self.wrapped_text_lines[self._cursor_pos[0]][-1]
                if self._cursor_pos[1] < len(wl):
                    self.set_cursor_pos(self._cursor_pos[0], len(wl))
                else:
                    self.set_cursor_pos(len(self.wrapped_text_lines) - 1, len(self.wrapped_text_lines[-1][-1]))
        else:
            if len(key) > 1:
                self.app.log("KEY", repr(key))
                return False
            else:
                self.insert_text_at_cursor(key)

        return True

    def _find_next_text_cursor_breakpoint(self, ofs: int, direction: int = 1):
        def _get_type(ch: str):
            if ch.isspace():
                return "space"
            if ch.isalpha():
                return "alpha"
            if ch.isnumeric():
                return "num"
            if ch in "-_:;,.!?":
                return "break"
            return "any"

        if 0 <= ofs < len(self.text):
            start_type = _get_type(self.text[ofs])
            while 0 <= ofs < len(self.text):
                ofs += direction
                if 0 <= ofs < len(self.text):
                    cur_type = _get_type(self.text[ofs])
                    if cur_type != start_type:
                        break
        return max(0, min(len(self.text), ofs))

    def insert_text_at_cursor(self, text: str):
        wrapped_lines = self.wrapped_text_lines
        if 0 <= self._cursor_pos[0] < len(wrapped_lines):
            y, x, line = wrapped_lines[self._cursor_pos[0]]
            offset = self.text_lines[y][0] + self._cursor_pos[1]
            self.set_text(self.text[:offset] + text + self.text[offset:])
            self.set_cursor_text_offset(offset + len(text))

        elif self._cursor_pos[0] >= len(wrapped_lines):
            if not self._text:
                self.set_text(text)
            else:
                new_lines = "\n" * (self._cursor_pos[0] - len(wrapped_lines) + 1)
                self.set_text(self._text + new_lines + text)
            self.set_cursor_text_offset(len(self.text))

    def render(self):
        if self._cursor_pos[0] < self._line_offset:
            self._line_offset = self._cursor_pos[0]
        if self._cursor_pos[0] - self._line_offset >= self.content_size[0]:
            self._line_offset = max(0, self._cursor_pos[0] - self.content_size[0] + 1)

        super().render()

        if self.is_focus:
            x = self.content_pos[1] + self._cursor_pos[1]
            y = self.content_pos[0] + min(self.content_size[0] - 1, self._cursor_pos[0] - self._line_offset)
            self.window.move(y, x)

    def move_cursor(self, offset: int):
        if offset < 0:
            while offset:
                if self._cursor_pos[1] > 0:
                    self._cursor_pos = (self._cursor_pos[0], self._cursor_pos[1] - 1)
                else:
                    if 0 < self._cursor_pos[0]:
                        self._cursor_pos = (
                            self._cursor_pos[0] - 1,
                            self.end_of_wrapped_line(self._cursor_pos[0] - 1)
                        )
                offset += 1
        else:
            while offset:
                if self._cursor_pos[1] < self.end_of_wrapped_line(self._cursor_pos[0]):
                    self._cursor_pos = (self._cursor_pos[0], self._cursor_pos[1] + 1)
                offset -= 1

    def set_cursor_pos(self, y: int, x: int):
        self._cursor_pos = (y, x)
        self.set_dirty()

    def get_cursor_text_offset(self) -> int:
        if self._cursor_pos[0] >= len(self.wrapped_text_lines):
            return len(self.text)
        org_y, off_x = self.wrapped_text_lines[self._cursor_pos[0]][:2]
        return self.text_lines[org_y][0] + off_x + self._cursor_pos[1]

    def set_cursor_text_offset(self, offset: int):
        o = 0
        for y, (org_y, x, line) in enumerate(self.wrapped_text_lines):
            next_o = o + len(line) + 1
            # self.app.log(f"of={offset}", y, x, o, next_o)
            if o <= offset < next_o:
                # self.app.log("set", y, offset - o)
                self.set_cursor_pos(y, offset - o)
                break
            o = next_o
