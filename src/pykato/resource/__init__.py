import os
import sys

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(RESOURCE_DIR, filename)


FONT_PROGGY_CLEAN = get_path("ProggyClean.ttf")
FONT_DONGLE_LIGHT = get_path("Dongle-Light.ttf")

