import os
import cv2
import numpy as np


B_THRESHOLD = 35  # threshold for increasing brightness


def preprocess(video_in: str):
    """
    Check video quality and apply preprocess if required,


    Returns
    -------
    name : str
        name of the processed video to continue analyzing

    """
    name = video_in

    # brightness
    if brightness_check(video_in):
        print(f"Increasing brightness for '{video_in}'")
        name = video_in[:-4] + '_bright' + '.mp4'
        if os.path.exists(name):
            print(f"Preprocessed video '{name}' exists, removing it")
            os.remove(name)
        increase_exposure(video_in, name)

    # TODO: anything else?

    return name


def map_video(
    func,
    video_in: str,
    video_out: str,
    fourcc: str = 'mp4v',
    fps: int = 25,
    dim=(640,
         480)):
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

    cap = cv2.VideoCapture(video_in)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(video_out, fourcc, fps, dim)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frame = func(frame)
            writer.write(frame)
        else:
            break

    cap.release()
    writer.release()


# brightness related functions
# ---------------------------------------------------------

# functions for checking brightness
def brightness_check(video: str):
    return (np.median(get_brightness(video)) <= B_THRESHOLD)


def get_brightness(video: str):

    # read 1st frame
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()

    # format into 1d array of all pixel brightness
    # (of 1st channel, this is GRAYSCALE ONLY)
    dist = frame[:, :, 0].flatten()
    return dist


# functions for increasing brightness
def exposure_table(gamma: float):
    gamma_table = np.power(np.arange(256) / 255, gamma) * 255
    gamma_table = np.round(gamma_table).astype(np.uint8)
    return gamma_table


def lut(frame, lut_table): return cv2.LUT(frame, lut_table)


def increase_exposure(video_in: str, video_out: str, value: float = 0.5):
    """Increase exposure of the given [video]"""
    lut_table = exposure_table(value)
    map_video(lambda x: lut(x, lut_table), video_in, video_out)

# ---------------------------------------------------------
