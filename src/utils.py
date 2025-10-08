# src/utils.py
import logging
import sys

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
