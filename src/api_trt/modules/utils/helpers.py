import os
from itertools import chain, islice
from distutils import util

def prepare_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def tobool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False

def parse_size(size=None, def_size='640,480'):
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst


def colorize_log(string, color):
    colors = dict(
        grey="\x1b[38;21m",
        yellow="\x1b[33;21m",
        red="\x1b[31;21m",
        bold_red="\x1b[31;1m",
    )
    reset = "\x1b[0m"
    col = colors.get(color)
    if col is None:
        return string
    string = f"{col}{string}{reset}"
    return string