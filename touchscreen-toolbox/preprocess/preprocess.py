import cv2
import numpy as np


def get_exposure(gamma : float):
    gamma_table = np.power(np.arange(256)/255, gamma) * 255
    gamma_table = np.round(gamma_table).astype(np.uint8)
    return gamma_table

def lut(frame, lut_table): return cv2.LUT(frame, lut_table)

def process(video_in : str, video_out : str, func, fourcc : str = 'mp4v', fps : int = 25, dim=(640, 480)):
    """
    Process video with the given [func]
    
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
    
    
def increase_exposure(video : str):
    """Increase exposure of the given [video]"""
    lut_table = get_exposure(0.5)
    process(video, video[:-4]+'_bright.mp4', lambda x: lut(x, increase))