import os
import cv2
import numpy as np
import touchscreen_toolbox.config as cfg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



def add_suffix(vid_info, suffix):
    """Add suffix (from preprocessing) to target video path"""
    return vid_info['target_path'][:-len(vid_info['format'])] + suffix + vid_info['format']


def preprocess_video(vid_info: dict):
    """
    Check video quality and apply preprocess if required,


    Returns
    -------
    video_in : str
        name of the processed video to continue analyzing

    """

    vid_info['prep'] = [] # to record preprocess applied
    
    cut_video(vid_info)
    
    brightness(vid_info)

    # TODO: anything else?


def cut_video(vid_info):
    """
    Cut video according to vid_info['time'] entry,
    *overwrite
    """
    
    print(f"Cutting '{vid_info['target_path']}'...")
    
    # retrieve time info
    try:
        start, end = vid_info['time']
    
    except KeyError:
        tb.utils.eprint(f"Time information not found for {vid_info['path']}")
        start, end = 0, vid_info['length']
    
    # cut & save to <target_path>
    vid_info['target_path'] = add_suffix(vid_info, '_c')
    ffmpeg_extract_subclip(vid_info['path'], start, end, targetname=vid_info['target_path'])
    
    vid_info['prep'].append('c')


def brightness(vid_info):
    """
    Bright preprocessing using gamma correction, if video brightness is too low
    (set brightness threshold in config)
    *overwrite
    """
    
    target_video = vid_info['target_path']
    
    # only apply preprocessing if the average brightness is lower than threshold
    if brightness_check(target_video):
        
        print(f"Increasing brightness for '{target_video}'...")
        
        vid_info['target_path'] = add_suffix(vid_info, '_b')

        gamma_correction(target_video, vid_info['target_path'], gamma=0.5)
        
        vid_info['prep'].append('b')
        
        os.remove(target_video)


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
    return (np.median(get_brightness(video)) <= cfg.B_THRESHOLD)


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
    
    # overwrite
    if os.path.exists(video_out):
        os.remove(video_out)
    
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