import os
import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

B_THRESHOLD = 40  # threshold for increasing brightness


def preprocess(video_in: str):
    """
    Check video quality and apply preprocess if required,


    Returns
    -------
    video_in : str
        name of the processed video to continue analyzing

    """
    name = video_in
    proc_ls = []

    # brightness
    if brightness_check(video_in):
        print(f"Increasing brightness for '{video_in}'")
        name = video_in[:-4] + '_bright' + '.mp4'
        if os.path.exists(name):
            print(f"Preprocessed video '{name}' exists, removing it")
            os.remove(name)
        gamma_correction(video_in, name, gamma=0.5)
        proc_ls.append('b')

    # TODO: anything else?

    return name, proc_ls


def clip_video(video_in: str, start: float, end: float, video_out: str=None):
    """Clip video"""
    ffmpeg_extract_subclip(video_in, start, end, targetname=video_out)
    

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
    
    # initialize opencv
    cap = cv2.VideoCapture(video_in)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(video_out, fourcc, fps, dim)
    
    # iterate each frame and apply function
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frame = func(frame)
            writer.write(frame)
        else:
            break

    cap.release()
    writer.release()


def lut(frame, lut_table): return cv2.LUT(frame, lut_table)


# brightness related functions
# ---------------------------------------------------------

FRAME2READ = 5

# functions for checking brightness
def brightness_check(video: str):
    return (np.median(get_brightness(video)) <= B_THRESHOLD)


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


# functions for gamma correction
def get_gamma_table(gamma: float):
    gamma_table = np.power(np.arange(256) / 255, gamma) * 255
    gamma_table = np.round(gamma_table).astype(np.uint8)
    return gamma_table


def gamma_correction(video_in: str, video_out: str, gamma: float = 0.5):
    """Increase exposure of the given [video]"""
    lut_table = get_gamma_table(gamma)
    map_video(lambda x: lut(x, lut_table), video_in, video_out)
    
#     # moviepy 
#     # somehow this is slower
#     clip = VideoFileClip(video_in)
#     clip2 = clip.fx(vfx.gamma_corr, gamma=gamma)
#     clip2.write_videofile(video_out)
#     clip.close()
#     clip2.close()

# ---------------------------------------------------------

P_SUFIXX = ['bright.mp4']  # possible preprocessed video suffix

def is_preprocess(name):
    return np.any([name.endswith(suffix) for suffix in P_SUFIXX])