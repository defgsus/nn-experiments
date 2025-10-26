small, hackish, static site renderer for the `docs/` directory.

in project root, call 
```shell
python scripts/blog/ render
```

this will update `docs/logs/README.md` file and the whole `docs/html` directory.

Result goes online to [defgsus.github.io/nn-experiments/](https://defgsus.github.io/nn-experiments/)


## Development

start a http server in the `docs/` directory, or even better, to match the public url:

```shell
# create a symlink to docs/ 
ln -s docs/ nn-experiments

# run server
python -m http.server
```

And visit http://localhost:8000/nn-experiments/


## Markdown specialities

### integrated footnotes

A footnote can be inserted likes this[[[start this is the footnote content end]]] and then the text continues.

However, the generated html/css is quite hacky. You can not put any block tag or another `<p>` inside the footnote
because the browser will move it outside the p-tag where the footnote reference is placed and then it will be 
always visible. The renderer will emit a warning if you do.

You can still make markdown paragraphs and code blocks. They are rendered inside a`<span style="white-space: pre-wrap">`. 
See `scripts.blog.render.HTMLInlineRenderer`.