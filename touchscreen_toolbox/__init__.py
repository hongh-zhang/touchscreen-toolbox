import sys
import logging
from .core import *
from . import utils
from . import extract
from . import postprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename='log.log')
stdout_handler = logging.StreamHandler(sys.stdout)
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)