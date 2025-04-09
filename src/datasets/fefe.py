import os
import time
from pathlib import Path
import datetime
from html.parser import HTMLParser
from typing import Optional

import requests
from tqdm import tqdm

from src import config
from src.datasets.base_iterable import BaseIterableDataset


FEFE_DATASET_PATH = config.SMALL_DATASETS_PATH / "fefe"


def iter_fefe_htmls(
        CACHE_PATH: Path = FEFE_DATASET_PATH / "html",
):
    ses = requests.session()
    ses.headers["User-Agent"] = "Hallo Fefe! Python 3.8 / requests 2.31"
    mindate = datetime.date(2005, 3, 1)
    maxdate = datetime.date.today()
    os.makedirs(CACHE_PATH, exist_ok=True)
    for year in range(2005, maxdate.year + 1):
        for month in range(1, 13):
            if year == mindate.year and month < mindate.month:
                continue
            if year == maxdate.year and month > maxdate.month:
                break

            filename = CACHE_PATH / "{:04d}-{:02d}.html".format(year, month)
            if not filename.exists():
                url = "https://blog.fefe.de/?mon={:04d}{:02d}".format(year, month)
                time.sleep(1)
                res = ses.get(url)
                filename.write_text(res.text)

            yield filename.read_text()


def iter_fefe_posts(
        CACHE_PATH: Path = FEFE_DATASET_PATH / "html",
):
    for markup in iter_fefe_htmls(CACHE_PATH=CACHE_PATH):
        yield from iter_posts_from_fefe_html(markup)


# --------------------- parsing --------------------------

MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

AUTO_CLOSE_TAGS = {
    "ul": ["ul", "li", "p", "b", "blockquote"],
    "li": ["li", "p", "b", "blockquote"],
    "p": ["p"],
    "blockquote": ["blockquote", "p"],
}


def _fefe_date_str_to_date(s):
    s = s.split()
    return datetime.datetime(int(s[-1]), MONTH_NAMES.index(s[-3])+1, int(s[-2]), 0, 0, 0)


def dump_post(post):
    print(post)
    for tag in post["tags"]:
        print(tag["tag"], "[%s]" % post["text"][tag["i"][0]:tag["i"][1]], tag.get("url", ""))


class PostParser(HTMLParser):

    def __init__(self, *args, **kwargs):
        super(PostParser, self).__init__(*args, **kwargs)
        self.tags = []
        self.cur_href = None
        self.day_index = None
        self.post = {
            "text": "",
            "tags": [],
        }

    def tag(self):
        return self.tags[-1] if self.tags else ""

    def push_tag(self, tag):
        self.tags.append(tag)

    def pop_tag(self):
        if self.tags:
            self.tags.pop()

    def inside_tag(self, tag):
        for t in reversed(self.tags):
            if t == tag:
                return True
        return False

    def tag_count(self, tag):
        return sum(1 for t in self.tags if t == tag)

    def handle_starttag(self, tag, attrs):
        #print("starttag", tag, self.tags)
        if tag == "a":
            attrs = {t[0]: t[1] for t in attrs}
            self.cur_href = attrs.get("href")

        for auto_close in AUTO_CLOSE_TAGS:
            if tag == auto_close:
                options = AUTO_CLOSE_TAGS[tag]
                while self.tag() in options:
                    self.handle_endtag(self.tag())
        if tag == "p" or tag == "br":
            if self.post:
                self.post["text"] += "\n"
            return
        #print("OPEN", tag, self.tags)
        self.push_tag(tag)

    def handle_endtag(self, tag):
        #print("CLOSE", tag, self.tags)
        if self.tags:
            if self.tags[-1] == tag:
                self.pop_tag()
            else:
                while True:
                    if self.tags[-1] in AUTO_CLOSE_TAGS.get(tag, []):
                        prevtag = self.tags[-1]
                        self.pop_tag()
                        if not self.tags or prevtag == tag:
                            break
                    else:
                        self.pop_tag()
                        break

    def handle_data(self, data):
        start_space = data.startswith(" ")
        end_space = data.endswith(" ")
        if start_space and self.post["text"] and not self.post["text"].endswith(" "):
            self.post["text"] += " "
        start = len(self.post["text"])
        self.post["text"] += data.replace("\n", " ").replace("  ", " ").strip()
        if end_space:
            self.post["text"] += " "
        end = len(self.post["text"])

        if self.tag():
            e = {"tag":self.tag(), "i": [start, end]}
            if self.tag() == "a" and self.cur_href:
                e["url"] = self.cur_href

            #if self.enclosing_tags:
            #    self.enclosing_tags[-1]["i"][1] = e["i"][1]
            #else:
            #    if e["tag"] in ENCLOSING_TAGS:
            #        self.enclosing_tags.append(e)

            for other in self.post["tags"]:
                if other["tag"] == e["tag"]:
                    if other["i"][0] <= e["i"][0] and other["i"][1] >= e["i"][1]:
                        other["i"][1] = e["i"][1]
                        e = None
                        break
            if e:
                self.post["tags"].append(e)


def iter_posts_from_fefe_html(markup):
    #markup = markup.split("""Siehe auch: <a href="//alternativlos.org/">Alternativlos</a><p>""")[1]

    import re
    days = []
    for match in re.finditer(r'<h3>(.+)</h3>', markup):
        days.append((match.span()[0], _fefe_date_str_to_date(match.groups()[0])))

    for i in range(len(days)):
        if i < len(days)-1:
            markup_part = markup[days[i][0]:days[i+1][0]]
        else:
            markup_part = markup[days[i][0]:]
        posts_markup = []
        for match in re.findall(r'<a href="(\?ts=[0-9a-z]+)">\[l\]</a>(.+)', markup_part):
            url = match[0]
            post_markup = match[1][1:]
            posts_markup.append((url, post_markup))

        for j, post_markup in enumerate(posts_markup):
            p = PostParser()
            p.feed(post_markup[1])
            post = p.post
            post["date"] = days[i][1]
            post["url"] = post_markup[0]
            post["day_index"] = len(posts_markup)-1-j

            yield post


class FefePostIterableDataset(BaseIterableDataset):

    def __init__(
            self,
    ):
        pass

    def __iter__(self):
        for post in iter_fefe_posts():
            yield post["text"]


if __name__ == "__main__":
    for post in tqdm(iter_fefe_posts()):
        #print(post["date"], len(post["text"]))
        #dump_post(post)
        pass
