# Chat LM 2

Is a chat bot again, this time based on the granite model (probably usable with others)
and some simple server/client implementation, to only reload the model occasionally.

`widgets.py` was an attempt to build a nice console UI via curses. It's quite nice but
actual text area editing is a nightmare to implement in spare time so i switched to
the browser once again.

```shell
# run server in separate console
python scripts/chat_lm2/server.py

# run text UI
python scripts/chat_lm2

# run browser UI (it just serves a html/js page)
python scripts/chat_lm2/browser.py
```
