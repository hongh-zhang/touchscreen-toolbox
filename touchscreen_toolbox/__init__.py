import sys
import logging
from .core import *
from .video_info import *
from . import utils
from . import postprocess
from . import pose_estimation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename="log.log")
erro_handler = logging.FileHandler(filename="err.log")
erro_handler.setLevel(40)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(20)

formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
erro_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(erro_handler)
logger.addHandler(stdout_handler)

logger.debug("Initialized")