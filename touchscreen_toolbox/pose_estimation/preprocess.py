import os
import cv2
import logging
import numpy as np
import touchscreen_toolbox.config as cfg
import ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

logger = logging.getLogger(__name__)


def preprocess_video(vid_info: dict):
    """
    Check video quality then apply preprocessing if required,
    """

    vid_info["prep"] = []  # to record preprocess applied
    resolution(vid_info)
    brightness(vid_info)


# --- functions for video resolution ---

def resolution(vid_info):
    width = cfg.RESOLUTION['width']
    height = cfg.RESOLUTION['height']
    target_video = vid_info["target_path"]
    probe_info = get_ffprobe_info(target_video)

    if (probe_info['height'] != height) or (probe_info['width'] != width):
        logger.info(f"Rescaling '{target_video}'...")

        vid_info["target_path"] = add_suffix(vid_info, "_r")
        vid_info["prep"].append("r")

        resize_video(target_video, vid_info["target_path"], probe_info, width, height)


def get_ffprobe_info(video_path):
    """Extract video information using ffprobe"""
    probe = ffmpeg.probe(video_path)
    return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)


def resize_video(video_path, output_path, probe_info, width, height):
    """Resize video resolution"""
    video = ffmpeg.input(video_path)
    # probe = ffmpeg.probe(video_path)
    video.filter('scale', width=width, height=height).output(output_path, video_bitrate=probe_info['bit_rate']).run()


# -------

def map_video(
        func,
        video_in: str,
        video_out: str,
        fourcc: str = "mp4v",
        fps: int = 25,
        dim=(640, 480),
):
    """
    Map video with the given [func]

    Args
    --------
    video_in : str
        path to input video

    video_out : str
        path to output video

    func : (:: ndarray -> ndarray)
        function to be mapped to each frame

    """

    try:
        # initialize opencv
        cap = cv2.VideoCapture(video_in)
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(video_out, fourcc, fps, dim)

        # iterate each frame and apply function
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame = func(frame)
                writer.write(frame)
            else:
                break
    except Exception as exc:
        raise exc
    # release file before terminating
    finally:
        try:
            cap.release()
            writer.release()
        except NameError:
            pass


def lut(frame, lut_table):
    return cv2.LUT(frame, lut_table)


# brightness related functions
# ---------------------------------------------------------

FRAME2READ = 5


# functions for checking brightness
def brightness_check(video: str):
    return np.median(get_brightness(video)) <= cfg.B_THRESHOLD


def get_brightness(video: str):
    # read 1st frame
    cap = cv2.VideoCapture(video)
    for _ in range(FRAME2READ):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()

    # format into 1d array of all pixel brightness
    # (of 1st channel, this is GRAYSCALE ONLY)
    dist = frame[:, :, 0].flatten()
    return dist


def brightness(vid_info):
    """
    Bright preprocessing using gamma correction, if video brightness is too low
    (set brightness threshold in config)
    """

    target_video = vid_info["target_path"]

    # only apply preprocessing if the average brightness is lower than threshold
    if brightness_check(target_video):
        logger.info(f"Increasing brightness for '{target_video}'...")

        vid_info["target_path"] = add_suffix(vid_info, "_b")

        gamma_correction(target_video, vid_info["target_path"], gamma=0.5)

        vid_info["prep"].append("b")


# functions for gamma correction
def get_gamma_table(gamma: float):
    gamma_table = np.power(np.arange(256) / 255, gamma) * 255
    gamma_table = np.round(gamma_table).astype(np.uint8)
    return gamma_table


def gamma_correction(video_in: str, video_out: str, gamma: float = 0.5):
    """Increase exposure of the given [video]"""

    # overwrite
    if os.path.exists(video_out):
        os.remove(video_out)

    lut_table = get_gamma_table(gamma)
    map_video(lambda x: lut(x, lut_table), video_in, video_out)


# ---------------------------------------------------------

P_SUFFIX = ["_b.mp4", "_c.mp4"]  # possible preprocessed video suffix


def is_preprocess(name):
    return np.any([name.endswith(suffix) for suffix in P_SUFFIX])


def add_suffix(vid_info, suffix):
    """Add suffix (from preprocessing) to target video path"""
    return (
            vid_info["target_path"][: -len(vid_info["format"])]
            + suffix
            + vid_info["format"]
    )


def cut_video(vid_info):
    """
    Cut video according to vid_info['time'] entry,
    """

    logger.info(f"Cutting '{vid_info['target_path']}'...")

    # retrieve time info
    try:
        start, end = vid_info["time"]

        # cut & save to <target_path>
        target = add_suffix(vid_info, "_c")
        ffmpeg_extract_subclip(vid_info["path"], start, end, targetname=target)

        vid_info["prep"].append("c")
        vid_info["target_path"] = target

    except KeyError:
        logger.warning(f"Time information not found for {vid_info['path']}")
