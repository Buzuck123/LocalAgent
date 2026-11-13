"""aiohttp wrapper with environment proxy support enabled by default."""

from aiohttp import *
from aiohttp import ClientSession as _ClientSession


class ClientSession(_ClientSession):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        super().__init__(*args, **kwargs)
