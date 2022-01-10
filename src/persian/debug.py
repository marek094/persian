import os

MAGIC = '__persdbg__'


def is_debug():
    return MAGIC in os.environ
